# SynapseGraph -- Sequential Implementation Prompts

> Each prompt below is designed to be executed in order. Every prompt declares its
> **prerequisites** (which prior prompts must be complete), the **ADRs/DDDs** it
> implements, the **crates/packages** it introduces, and the **artifacts** it
> produces that later prompts depend on.
>
> Do NOT skip prompts or reorder them. The dependency chain is strict.

---

## Prompt 01 -- Workspace Scaffolding & Shared Domain Types

**Prerequisites:** None (first prompt)
**Implements:** ADR-007 (ruvector ecosystem dependency selection), ADR-006 (WASM-first, workspace structure)
**Introduces:** `ruvector-core` v2.0.5, `ruvector-math` v2.0.4, `ruvector-collections` v2.0.4 (Layer 0 crates)

### Prompt

```
Set up the SynapseGraph Rust workspace. Create a Cargo workspace at the repo root
with the following member crates:

  synapse-graph-types    -- shared domain types used across all bounded contexts
  synapse-graph-core     -- application orchestration (will wire domains together later)
  synapse-graph-wasm     -- WASM compilation target (empty shell for now)

In synapse-graph-types, define the foundational value objects that every domain
depends on. Reference the ubiquitous language tables in DDD-001 through DDD-007.
At minimum define:

  - EmbeddingId, ChannelId, SessionId, NodeId, ClusterId, ShortcutId,
    PredictionId, EventId, PatternId, IntentId, LayerId  (newtype wrappers)
  - Timestamp (u64 microseconds since epoch)
  - SpatialPosition { x: f32, y: f32, z: f32 }
  - TemporalTier enum { Fast, Medium, Slow }
  - TemporalMeta { tier, timestamp, channel_id, session_id, coherence_score }
  - SpikeEvent { channel_id, amplitude: f32, sub_timestamp: Timestamp }
  - SpikeBatch { timestamp, session_id, spikes: Vec<SpikeEvent>, duration_us: u32 }
  - DenyReason enum (ManifoldOutlier, TemporalDiscontinuity, ChannelUnhealthy,
    SaturatedSignal, ZeroSignal) -- from DDD-001
  - CoherenceVerdict enum { Permit { score: f32 }, Deny { reason: DenyReason } }

All types must derive Clone, Debug, Serialize, Deserialize (serde).
All ID types must derive Copy, Eq, Hash.
The crate must be no_std compatible (use alloc where needed) per ADR-006.

Add ruvector-core = "2.0.5", ruvector-math = "2.0.4", ruvector-collections = "2.0.4"
as workspace-level dependencies. synapse-graph-types should not depend on them yet --
it is a pure domain types crate.

Also create a package.json at the repo root with:
  ruvector: ^0.2.11
  @ruvector/gnn: ^0.1.25
  @ruvector/rvf-wasm: ^0.1.6

Write unit tests that construct each type and round-trip through serde_json.
```

### Produces

- `synapse-graph-types` crate with all shared ID types, enums, and value objects
- Cargo workspace with three member crates
- package.json with npm dependencies
- All later prompts import from `synapse-graph-types`

---

## Prompt 02 -- GatedEmbedding & Coherence Gate Module

**Prerequisites:** Prompt 01 (shared types)
**Implements:** ADR-002 (coherence gating hard boundary), DDD-001 (CoherenceGate, GatedEmbedding)
**Introduces:** `cognitum-gate-kernel` v0.1.1, `ruvector-math` v2.0.4

### Prompt

```
Create the crate synapse-graph-ingestion as a new workspace member.

Implement the CoherenceGate module per DDD-001 and ADR-002. This is the hard
architectural boundary that prevents any raw embedding from reaching downstream
modules.

1. Define GatedEmbedding in synapse-graph-types:
     pub struct GatedEmbedding {
         vector: Vec<f32>,
         channel_id: ChannelId,
         timestamp: Timestamp,
         session_id: SessionId,
         coherence_score: f32,
     }
   Make the struct's fields private. Provide only pub getter methods.
   The ONLY constructor must be a pub(crate) fn inside synapse-graph-ingestion
   so no other crate can fabricate a GatedEmbedding. This enforces the type-level
   guarantee from ADR-002.

2. Define ChannelHealth { impedance: f32, snr: f32, dropout_rate: f32 } in types.

3. In synapse-graph-ingestion::gate, implement:
     pub struct CoherenceGate {
         config: GateConfig,
         channel_stats: HashMap<ChannelId, RunningManifoldStats>,
     }
     impl CoherenceGate {
         pub fn evaluate(&mut self, raw: &[f32], channel: ChannelId,
                         timestamp: Timestamp, session: SessionId)
             -> CoherenceVerdict;
         pub(crate) fn mint_gated(&self, raw: Vec<f32>, channel: ChannelId,
                                   timestamp: Timestamp, session: SessionId,
                                   score: f32) -> GatedEmbedding;
     }

   The evaluate() method performs:
     a) Manifold consistency check: Mahalanobis distance from running mean/covariance
        using ruvector-math. Reject if > config.sigma_threshold (default 3.0).
     b) Temporal continuity: compare vs last embedding from same channel.
        Reject if L2 delta exceeds config.max_temporal_delta.
     c) Channel health: reject if channel SNR < config.min_snr or
        dropout_rate > config.max_dropout.
     d) Zero/saturated signal: reject if all values are 0 or all are at max float.

   Use cognitum-gate-kernel for the no_std gate logic if it provides a suitable
   trait; otherwise wrap its primitives.

4. Define domain events in synapse-graph-types:
     EmbeddingIngested { embedding_id, gated_embedding (cloned), btsp_metadata }
     EmbeddingDenied { embedding_id, reason: DenyReason, channel_id, timestamp }
     ChannelHealthChanged { channel_id, previous, current }

5. Write tests:
   - A valid embedding passes the gate and returns Permit.
   - An outlier embedding (Mahalanobis > 3 sigma) is denied with ManifoldOutlier.
   - A zero-signal embedding is denied with ZeroSignal.
   - Construct GatedEmbedding only through the gate; verify that external crates
     cannot construct it (this is a compile-time test -- document it).
   - Benchmark: evaluate() completes in < 1 microsecond for dim=512.
```

### Produces

- `synapse-graph-ingestion` crate with `CoherenceGate`
- `GatedEmbedding` type-safe constructor pattern
- Coherence domain events
- All downstream prompts receive only `GatedEmbedding`, never raw vectors

---

## Prompt 03 -- Nervous System Processor & Spike Ingester

**Prerequisites:** Prompt 02 (CoherenceGate, GatedEmbedding)
**Implements:** DDD-001 (SpikeIngester aggregate, NervousSystemProcessor, BCI Adapter ACL)
**Introduces:** `ruvector-nervous-system` v2.0.4, `ruvector-nervous-system-wasm` v2.0.4

### Prompt

```
Extend synapse-graph-ingestion to implement the full SpikeIngester aggregate root
per DDD-001.

1. Implement NervousSystemProcessor wrapping ruvector-nervous-system:
     pub struct NervousSystemProcessor {
         inner: ruvector_nervous_system::SpikingNetwork,
         config: NervousSystemConfig,
     }
     impl NervousSystemProcessor {
         pub fn process_spikes(&mut self, batch: &SpikeBatch)
             -> ProcessedBatch;
     }
   ProcessedBatch contains:
     embeddings: Vec<RawEmbedding>  (RawEmbedding = { vector: Vec<f32>, channel_id })
     btsp_metadata: BtspMetadata    (learning rule outputs -- define in types)
     timing: TimingMetadata         (inter-spike intervals, burst detection flags)

   Enable BTSP learning on every call per DDD-001 invariant 4: "BTSP learning is
   always active."

2. Implement the SpikeIngester aggregate root:
     pub struct SpikeIngester {
         nervous_system: NervousSystemProcessor,
         coherence_gate: CoherenceGate,   // from Prompt 02
         channel_registry: HashMap<ChannelId, Channel>,
     }
     impl SpikeIngester {
         pub fn ingest(&mut self, batch: SpikeBatch) -> IngestionResult;
     }
   IngestionResult contains:
     gated: Vec<GatedEmbedding>
     denied: Vec<EmbeddingDenied>
     channel_health_changes: Vec<ChannelHealthChanged>

   The ingest() method follows the SPARC pseudocode:
     a) nervous_system.process_spikes(batch) -> ProcessedBatch
     b) For each raw embedding: coherence_gate.evaluate()
        - PERMIT -> mint_gated(), add to gated list
        - DENY -> create EmbeddingDenied event, add to denied list
     c) Update channel health stats; emit ChannelHealthChanged if thresholds crossed

3. Define the BCI Adapter anti-corruption layer trait:
     pub trait BciAdapter: Send + Sync {
         fn translate(&self, raw_data: &[u8]) -> Result<SpikeBatch, AdapterError>;
         fn channel_map(&self) -> &[ChannelId];
         fn sampling_rate_hz(&self) -> u32;
     }
   Provide a MockBciAdapter that generates synthetic spike batches with
   configurable patterns (constant, burst, drift, noise). This is the spike-train
   simulator from SPARC Phase 1 Task 6.

4. Write tests:
   - Ingest a clean SpikeBatch -> all embeddings appear in gated list.
   - Ingest a SpikeBatch with injected noise -> noisy channels are denied.
   - MockBciAdapter produces syntactically correct SpikeBatches.
   - Channel health transitions emit ChannelHealthChanged events.
   - Benchmark: full ingest() for 1024-channel batch completes in < 5ms.
```

### Produces

- Complete `SpikeIngester` aggregate root
- `NervousSystemProcessor` wrapping `ruvector-nervous-system`
- `BciAdapter` trait + `MockBciAdapter` for testing
- `IngestionResult` containing `Vec<GatedEmbedding>` -- this is the primary input for Prompts 04 and 05

---

## Prompt 04 -- Vector Memory (HNSW Store + Collections)

**Prerequisites:** Prompt 02 (GatedEmbedding type)
**Implements:** DDD-002 (VectorMemory aggregate, HnswStore, Collection)
**Introduces:** `ruvector-core` v2.0.5, `ruvector-collections` v2.0.4, `ruvector-wasm` v0.1.29

### Prompt

```
Create the crate synapse-graph-memory as a new workspace member.

Implement the VectorMemory aggregate root per DDD-002. This is the persistent
storage substrate that all later domains query.

1. Implement HnswStore wrapping ruvector-core:
     pub struct HnswStore {
         index: ruvector_core::HnswIndex,
         dimension: usize,
         config: HnswConfig,  // ef_construction, ef_search, m
     }
   Provide insert(), search_knn(query, k), and total_count().

2. Implement VectorMemory aggregate root:
     pub struct VectorMemory {
         store: HnswStore,
         collections: CollectionManager,  // wraps ruvector-collections
         metadata: HashMap<EmbeddingId, TemporalMeta>,
     }
     impl VectorMemory {
         pub fn insert(&mut self, embedding: GatedEmbedding, meta: TemporalMeta)
             -> Result<EmbeddingId, InsertError>;
         pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
         pub fn find_temporally_adjacent(&self, cluster: &EmbeddingCluster,
                                          window: Duration) -> Vec<TemporalNeighbor>;
         pub fn cluster_recent(&self, threshold: f32) -> Vec<EmbeddingCluster>;
     }

   The insert() method:
     a) Accepts only GatedEmbedding (compile-time enforcement from ADR-002).
     b) Assigns an EmbeddingId.
     c) Inserts vector into HnswStore.
     d) Stores TemporalMeta in the metadata map.
     e) Returns EmbeddingId.

   Define SearchResult { embedding_id, vector, distance, metadata: TemporalMeta }.
   Define EmbeddingCluster { centroid, member_ids, metadata, temporal_span }.
   Define TemporalNeighbor { node, correlation }.

3. Implement CollectionManager wrapping ruvector-collections:
     pub struct CollectionManager { ... }
     impl CollectionManager {
         pub fn create(&mut self, name: &str, filter: CollectionFilter) -> CollectionId;
         pub fn list(&self) -> Vec<Collection>;
     }

4. Define domain events:
     EmbeddingStored { embedding_id, collection_id, metadata, hnsw_neighbors }
     EmbeddingsEvicted { collection_id, count, reason }

5. Implement cluster_recent():
   Use the HNSW neighborhood structure to group recent embeddings into clusters.
   Return EmbeddingCluster values that the Cognitive Graph domain (Prompt 08) will
   consume.

6. Implement find_temporally_adjacent():
   Given a cluster, find all embeddings from other clusters that fired within the
   specified time window. Return TemporalNeighbor with correlation scores.
   The Cognitive Graph domain (Prompt 08) uses this to create co-activation edges.

7. Write tests:
   - Insert a GatedEmbedding, search_knn returns it as the nearest neighbor.
   - Insert 1000 embeddings with known clusters, cluster_recent() finds them.
   - find_temporally_adjacent() correctly identifies co-activated embeddings.
   - Collections partition the store correctly.
   - Benchmark: insert < 50us, search_knn(k=10) < 200us for 10k embeddings.
```

### Produces

- `synapse-graph-memory` crate with `VectorMemory` aggregate
- `search_knn()` API consumed by Intent Prediction (Prompt 10)
- `cluster_recent()` API consumed by Cognitive Graph (Prompt 08)
- `find_temporally_adjacent()` API consumed by Cognitive Graph (Prompt 08)
- `EmbeddingStored` events consumed by Provenance (Prompt 12)

---

## Prompt 05 -- Temporal Router & Fast Tier

**Prerequisites:** Prompt 02 (GatedEmbedding), Prompt 04 (TemporalMeta already defined)
**Implements:** ADR-001 (three-tier temporal, fast tier), DDD-003 (TemporalRouter, FastTier)
**Introduces:** `ruvector-router-core` v2.0.4, `ruvector-tiny-dancer-core` v2.0.4, `ruvector-tiny-dancer-wasm` v2.0.4

### Prompt

```
Create the crate synapse-graph-temporal as a new workspace member.

Implement the TemporalRouter and FastTier per DDD-003 and ADR-001. This prompt
builds the first tier; Prompts 06 and 07 add medium and slow tiers.

1. Implement TemporalRouter wrapping ruvector-router-core + ruvector-tiny-dancer-core:
     pub struct TemporalRouter {
         fastgrnn: TinyDancerModel,  // from ruvector-tiny-dancer-core
         config: RouterConfig,
     }
     impl TemporalRouter {
         pub fn classify(&self, embedding: &GatedEmbedding,
                          timing: &TimingMetadata) -> TemporalTier;
     }

   Classification logic:
     a) Examine inter-spike interval from timing metadata.
        If median ISI < 10ms -> TemporalTier::Fast
     b) Examine session position. If embedding shows session-level trend
        (compared to session baseline) -> TemporalTier::Medium
     c) Compare against slow-tier baseline (passed via config or lazy-loaded).
        If significant drift detected -> TemporalTier::Slow
     d) Default -> TemporalTier::Medium

   The FastGRNN model from ruvector-tiny-dancer-core provides the learned
   classifier. Initialize with default weights; the Adaptation domain (Prompt 11)
   will fine-tune later.

   Performance: classify() must complete in < 10 microseconds.

2. Implement FastTier:
     pub struct FastTier {
         buffer: RingBuffer<TimestampedEmbedding>,
         capacity: usize,
         pattern_detector: BurstPatternDetector,
     }
     impl FastTier {
         pub fn push(&mut self, embedding: &GatedEmbedding, timestamp: Timestamp);
         pub fn detect_persistent_patterns(&self) -> Vec<PersistentPattern>;
         pub fn recent_context(&self, window: Duration) -> Vec<&GatedEmbedding>;
     }

   RingBuffer is a fixed-capacity circular buffer. On overflow, the oldest entry
   is silently overwritten (FIFO eviction per DDD-003).

   BurstPatternDetector monitors for patterns persisting > 500ms with > 10
   activations. These are candidates for promotion to the medium tier (Prompt 06).

3. Define TimestampedEmbedding { embedding: GatedEmbedding (cloned), timestamp }.

4. Define PersistentPattern { pattern_id, centroid, activation_count, duration }.

5. Define domain event:
     TierEvicted { tier: TemporalTier::Fast, evicted_count, reason: RingBufferOverflow }

6. Write tests:
   - classify() returns Fast for spike-burst timing metadata (ISI < 10ms).
   - classify() returns Medium for session-level timing.
   - FastTier ring buffer correctly evicts oldest entries at capacity.
   - detect_persistent_patterns() finds a planted 500ms burst pattern.
   - Benchmark: classify() < 10us, push() < 1us.
```

### Produces

- `synapse-graph-temporal` crate with `TemporalRouter` and `FastTier`
- `classify()` API used by the orchestration layer (Prompt 13) to route each GatedEmbedding
- `PersistentPattern` output consumed by Medium Tier (Prompt 06) for promotion
- `recent_context()` used by TemporalContext blending (Prompt 07)

---

## Prompt 06 -- Medium Tier (Session Tensors) & Tier Promotion

**Prerequisites:** Prompt 05 (TemporalRouter, FastTier, PersistentPattern)
**Implements:** ADR-001 (medium tier), DDD-003 (MediumTier, promotion logic)
**Introduces:** `ruvector-temporal-tensor` v2.0.4

### Prompt

```
Extend synapse-graph-temporal with the MediumTier and tier promotion logic
per DDD-003 and ADR-001.

1. Implement MediumTier wrapping ruvector-temporal-tensor:
     pub struct MediumTier {
         current_session: SessionTensor,
         past_sessions: Vec<CompressedSessionTensor>,
         quantization_bits: u8,  // default 8
         max_past_sessions: usize,  // default 10
     }
     impl MediumTier {
         pub fn accumulate(&mut self, embedding: &GatedEmbedding);
         pub fn extract_sequences(&self, cluster: &EmbeddingCluster)
             -> Vec<TemporalSequence>;
         pub fn session_context(&self) -> Vec<f32>;
         pub fn end_session(&mut self);
         pub fn cross_session_stability(&self, pattern: &PersistentPattern) -> f32;
     }

   SessionTensor uses ruvector-temporal-tensor with 8-bit quantization.
   On end_session(): compress current session to CompressedSessionTensor,
   push to past_sessions, evict oldest if > max_past_sessions.

2. Define TemporalSequence per DDD-003/DDD-004:
     pub struct TemporalSequence {
         pub from_cluster: ClusterId,
         pub to_cluster: ClusterId,
         pub confidence: f32,
         pub observed_count: u32,
         pub average_delay: Duration,
     }

   extract_sequences() examines the session tensor for ordered cluster pairs
   that appear consistently. The Cognitive Graph (Prompt 08) uses these to
   create TEMPORAL_SEQUENCE edges.

3. Implement tier promotion logic in a new module synapse-graph-temporal::promotion:
     pub struct TierPromoter {
         fast_to_medium_threshold: Duration,      // default 500ms
         fast_to_medium_min_activations: u32,     // default 10
         medium_to_slow_sessions: usize,          // default 3
     }
     impl TierPromoter {
         pub fn check_fast_promotions(&self, fast: &FastTier)
             -> Vec<PromotionCandidate>;
         pub fn check_medium_promotions(&self, medium: &MediumTier)
             -> Vec<PromotionCandidate>;
         pub fn execute_promotion(&self, candidate: PromotionCandidate,
                                   fast: &mut FastTier,
                                   medium: &mut MediumTier,
                                   slow: &mut SlowTier);  // SlowTier from Prompt 07
     }

   PromotionCandidate { pattern: PersistentPattern, from_tier, to_tier }.

   Promotion rules from DDD-003:
     Fast -> Medium: pattern persists > 500ms with > 10 activations
     Medium -> Slow: pattern stable across 3+ sessions
       (uses MediumTier::cross_session_stability() > 0.8)

4. Define domain event:
     PatternPromoted { pattern_id, from_tier, to_tier, persistence_duration,
                       activation_count }

5. Write tests:
   - accumulate() 100 embeddings, session_context() returns non-zero vector.
   - extract_sequences() finds planted A->B temporal patterns.
   - end_session() compresses and archives the session tensor.
   - TierPromoter detects a pattern persisting > 500ms in the fast tier.
   - Promotion from fast to medium moves the pattern correctly.
   - Benchmark: accumulate() < 50us, extract_sequences() < 5ms.
```

### Produces

- `MediumTier` with session tensor management
- `TemporalSequence` output consumed by Cognitive Graph (Prompt 08) for edge creation
- `TierPromoter` with fast-to-medium promotion (medium-to-slow wired in Prompt 07)
- `PatternPromoted` events consumed by Provenance (Prompt 12)

---

## Prompt 07 -- Slow Tier (Drift Vectors), TemporalLearner Aggregate & Context Blending

**Prerequisites:** Prompt 05 (FastTier, TemporalRouter), Prompt 06 (MediumTier, TierPromoter)
**Implements:** ADR-001 (slow tier, cross-tier context), DDD-003 (SlowTier, TemporalLearner aggregate root, TemporalContext)
**Introduces:** `ruvector-sona` v0.1.6 (first use -- slow tier consolidation)

### Prompt

```
Extend synapse-graph-temporal with the SlowTier and assemble the complete
TemporalLearner aggregate root per DDD-003.

1. Implement SlowTier:
     pub struct SlowTier {
         drift_vectors: Vec<DriftVector>,
         quantization_bits: u8,  // default 4
         baseline: Vec<f32>,     // long-term neural encoding baseline
         last_consolidation: Timestamp,
     }
     impl SlowTier {
         pub fn consolidate(&mut self, pattern: &PersistentPattern,
                             ewc_lambda: f32);
         pub fn detect_drift(&self, current: &[f32]) -> Option<DriftDetected>;
         pub fn baseline_context(&self) -> Vec<f32>;
         pub fn requantize(&mut self, target_bits: u8);  // 4-bit -> 2-bit under pressure
     }

   DriftVector { direction: Vec<f32>, magnitude: f32, detected_at: Timestamp,
                 affected_channels: Vec<ChannelId> }

   consolidate() uses ruvector-sona's EWC++ to absorb medium-tier patterns into
   the slow-tier baseline without destroying existing drift vectors. This is the
   first integration point for ruvector-sona; the full two-tier LoRA (Prompt 11)
   builds on this.

   detect_drift() compares a current embedding against the baseline using
   ruvector-math optimal transport distance. Returns DriftDetected if distance
   exceeds threshold.

2. Define DriftDetected event per DDD-003:
     DriftDetected { drift_magnitude, affected_channels, drift_direction }

3. Wire medium-to-slow promotion in TierPromoter::execute_promotion():
   When promoting from medium to slow, call slow.consolidate() with the
   pattern and EWC lambda from config.

4. Implement TemporalLearner aggregate root -- the unified entry point:
     pub struct TemporalLearner {
         router: TemporalRouter,     // Prompt 05
         fast: FastTier,             // Prompt 05
         medium: MediumTier,         // Prompt 06
         slow: SlowTier,             // this prompt
         promoter: TierPromoter,     // Prompt 06
         config: TemporalConfig,
     }
     impl TemporalLearner {
         pub fn accumulate(&mut self, embedding: &GatedEmbedding,
                            timing: &TimingMetadata);
         pub fn get_context(&self, weights: TierWeights) -> TemporalContext;
         pub fn extract_sequences(&self, cluster: &EmbeddingCluster)
             -> Vec<TemporalSequence>;
         pub fn maintain(&mut self) -> MaintenanceResult;
         pub fn end_session(&mut self);
     }

   accumulate():
     a) router.classify(embedding, timing) -> tier
     b) Match tier:
        Fast   -> fast.push(embedding, timestamp)
        Medium -> medium.accumulate(embedding)
        Slow   -> slow.consolidate(embedding.as_pattern(), ewc_lambda)

   get_context():
     TierWeights { fast: 0.5, medium: 0.3, slow: 0.2 } (configurable)
     Blend: fast.recent_context() * w_fast + medium.session_context() * w_medium
            + slow.baseline_context() * w_slow
     Return TemporalContext { fast_component, medium_component, slow_component, weights }

   maintain():
     a) promoter.check_fast_promotions(fast) -> execute if any
     b) promoter.check_medium_promotions(medium) -> execute if any
     c) Check memory budget; requantize slow tier if needed
     d) Return MaintenanceResult { promoted, evicted }

   end_session():
     medium.end_session()

5. Define TemporalContext per DDD-003:
     pub struct TemporalContext {
         pub fast_component: Vec<f32>,
         pub medium_component: Vec<f32>,
         pub slow_component: Vec<f32>,
         pub weights: TierWeights,
     }
     impl TemporalContext {
         pub fn blended(&self) -> Vec<f32>;
     }

   TierWeights { fast: f32, medium: f32, slow: f32 }.

6. Write tests:
   - Full TemporalLearner.accumulate() routes to correct tier.
   - get_context() blends three tiers with correct weights.
   - maintain() promotes a persistent fast-tier pattern to medium.
   - maintain() promotes a stable medium-tier pattern to slow.
   - Drift detection fires when a pattern diverges from baseline.
   - SlowTier.requantize() reduces quantization bits under pressure.
   - end_session() correctly archives the current session.
```

### Produces

- Complete `TemporalLearner` aggregate root with all three tiers
- `get_context()` API consumed by Intent Prediction (Prompt 10)
- `extract_sequences()` API consumed by Cognitive Graph (Prompt 08)
- `DriftDetected` event consumed by Adaptation (Prompt 11)
- `maintain()` drives the promotion/eviction lifecycle

---

## Prompt 08 -- Cognitive Graph (Hypergraph + Edge Construction)

**Prerequisites:** Prompt 04 (VectorMemory -- cluster_recent, find_temporally_adjacent), Prompt 06 (TemporalSequence)
**Implements:** ADR-005 (self-healing graph), DDD-004 (CognitiveGraph aggregate, HypergraphStore, edge construction rules)
**Introduces:** `ruvector-graph` v2.0.4, `ruvector-mincut` v2.0.4

### Prompt

```
Create the crate synapse-graph-cognitive as a new workspace member.

Implement the CognitiveGraph aggregate root (the hypergraph store, edge
construction, and self-healing) per DDD-004 and ADR-005. GNN shortcut discovery
is deferred to Prompt 09.

1. Implement HypergraphStore wrapping ruvector-graph:
     pub struct HypergraphStore {
         inner: ruvector_graph::HypergraphDb,
     }
     impl HypergraphStore {
         pub fn upsert_node(&mut self, centroid: Vec<f32>,
                             metadata: ClusterMetadata) -> NodeId;
         pub fn upsert_edge(&mut self, source: NodeId, target: NodeId,
                             edge_type: EdgeType, weight: f32);
         pub fn to_adjacency(&self) -> AdjacencyMatrix;
         pub fn node_features(&self) -> FeatureMatrix;
         pub fn nodes(&self) -> Vec<CognitiveNode>;
         pub fn edges(&self) -> Vec<CognitiveEdge>;
         pub fn neighbors(&self, node: NodeId) -> Vec<(NodeId, CognitiveEdge)>;
     }

2. Define CognitiveNode per DDD-004:
     pub struct CognitiveNode {
         pub id: NodeId,
         pub centroid: Vec<f32>,
         pub feature_vector: Vec<f32>,
         pub activation_count: u64,
         pub last_activated: Timestamp,
         pub first_seen: Timestamp,
         pub source_cluster: ClusterId,
     }

3. Define EdgeType enum and CognitiveEdge per DDD-004:
     pub enum EdgeType { CoActivation, TemporalSequence, SemanticSimilarity, Reinforcement }
     pub struct CognitiveEdge {
         pub source: NodeId, pub target: NodeId, pub edge_type: EdgeType,
         pub weight: f32, pub created_at: Timestamp,
         pub last_reinforced: Timestamp, pub reinforcement_count: u32,
     }

4. Implement graph update logic -- the update() method on CognitiveGraph:
     pub fn update(&mut self,
                   clusters: Vec<EmbeddingCluster>,      // from VectorMemory.cluster_recent()
                   sequences: Vec<TemporalSequence>,     // from TemporalLearner.extract_sequences()
                   temporal_neighbors: Vec<TemporalNeighbor>)  // from VectorMemory.find_temporally_adjacent()
         -> GraphUpdateResult;

   This method:
     a) For each cluster: upsert_node(centroid, metadata). If a node with similar
        centroid already exists (cosine similarity > 0.95), update it instead of
        creating a new one.
     b) For each temporal_neighbor: upsert_edge(CoActivation, weight=correlation).
     c) For each temporal_sequence: upsert_edge(TemporalSequence, weight=confidence).
     d) For nodes within top-k KNN of each other: upsert_edge(SemanticSimilarity,
        weight = 1.0 - cosine_distance).

5. Implement GraphHealer wrapping ruvector-mincut:
     pub struct GraphHealer {
         config: HealerConfig,  // min_cut_threshold, max_reinforcement_edges, prune_age
     }
     impl GraphHealer {
         pub fn analyze(&self, graph: &HypergraphStore) -> Vec<WeakPoint>;
         pub fn heal(&self, graph: &mut HypergraphStore,
                      weak_points: Vec<WeakPoint>,
                      historical_sequences: &[TemporalSequence])
             -> HealingResult;
     }

   WeakPoint { region_a, region_b, min_cut_value, bridge_edges } per DDD-004.

   heal():
     a) For each weak_point: search historical_sequences for evidence connecting
        region_a and region_b.
     b) If evidence exists: add Reinforcement edges.
     c) If no evidence: prune the bridge (it's spurious).
     d) Prune dead nodes (last_activated older than prune_age threshold).
     e) Return HealingResult { weak_points_reinforced, dead_regions_pruned,
        nodes_removed, edges_added, edges_removed }.

6. Assemble CognitiveGraph aggregate root:
     pub struct CognitiveGraph {
         hypergraph: HypergraphStore,
         healer: GraphHealer,
         shortcuts: ShortcutRegistry,  // simple HashMap<ShortcutId, CognitiveShortcut> for now
         config: CognitiveGraphConfig,
     }
   The GNN engine field will be added in Prompt 09.

7. Define domain events:
     GraphHealed { weak_points_reinforced, dead_regions_pruned, nodes_removed,
                   edges_added, edges_removed }
     GraphTopologyChanged { total_nodes, total_edges, average_min_cut, shortcut_count }

8. Write tests:
   - update() with 5 clusters creates 5 nodes and appropriate edges.
   - Duplicate clusters (cosine sim > 0.95) merge into existing nodes.
   - Co-activation edges are created for temporally adjacent clusters.
   - TemporalSequence edges are created from the temporal learner output.
   - GraphHealer detects a planted bottleneck (two subgraphs connected by 1 edge).
   - heal() reinforces the bottleneck with historical evidence.
   - heal() prunes dead nodes older than threshold.
   - Benchmark: update() with 100 clusters < 50ms, heal() < 100ms.
```

### Produces

- `synapse-graph-cognitive` crate with `CognitiveGraph` aggregate (minus GNN)
- `HypergraphStore` with node/edge CRUD
- `GraphHealer` with min-cut analysis and self-healing
- Graph consumes `EmbeddingCluster` from Prompt 04 and `TemporalSequence` from Prompt 06
- `ShortcutRegistry` (populated by GNN in Prompt 09, queried by Intent Prediction in Prompt 10)

---

## Prompt 09 -- GNN Engine & Cognitive Shortcut Discovery

**Prerequisites:** Prompt 08 (CognitiveGraph, HypergraphStore with to_adjacency/node_features)
**Implements:** ADR-003 (proof-gated graph attention), DDD-004 (GnnEngine, ShortcutDiscovery, CognitiveShortcut, AttentionProof)
**Introduces:** `ruvector-gnn` v2.0.5, `ruvector-graph-transformer` v2.0.4, `ruvector-gnn-wasm` v2.0.4, `ruvector-attention` v2.0.4

### Prompt

```
Extend synapse-graph-cognitive with the GNN engine and shortcut discovery
per DDD-004 and ADR-003.

1. Implement GnnEngine wrapping ruvector-gnn + ruvector-graph-transformer:
     pub struct GnnEngine {
         gnn: ruvector_gnn::GraphNeuralNetwork,
         transformer: ruvector_graph_transformer::ProofGatedTransformer,
         attention_config: AttentionModuleConfig,
     }
     impl GnnEngine {
         pub fn forward(&self, adjacency: &AdjacencyMatrix,
                         features: &FeatureMatrix)
             -> GnnOutput;
         pub fn extract_motifs(&self, output: &GnnOutput,
                                config: &MotifConfig)
             -> Vec<CognitiveShortcut>;
     }

   Configure the ProofGatedTransformer with three modules per ADR-003:
     - Temporal: attention weighted by recency. Proof = monotonic decrease with time.
     - Manifold: attention weighted by geodesic distance. Proof = manifold metric respected.
     - Biological: attention weighted by anatomical priors. Proof = consistent with
       connectivity matrix.

   Composition: final_attn = 0.5 * temporal + 0.3 * manifold + 0.2 * biological

   GnnOutput contains:
     node_embeddings: Vec<Vec<f32>>   -- updated node feature vectors
     attention_weights: AttentionWeights
     proofs: Vec<AttentionProof>      -- one per attention module per subgraph

2. Define CognitiveShortcut per DDD-004:
     pub struct CognitiveShortcut {
         pub id: ShortcutId,
         pub nodes: Vec<NodeId>,
         pub edges: Vec<CognitiveEdge>,
         pub activation_frequency: f32,
         pub predictive_power: f32,
         pub discovered_at: Timestamp,
         pub proof: AttentionProof,
         pub intent_associations: Vec<IntentAssociation>,
     }

   IntentAssociation { intent_id: IntentId, strength: f32 }.

3. Define AttentionProof per DDD-004:
     pub struct AttentionProof {
         pub temporal_proof: Vec<u8>,   // serialized proof from temporal module
         pub manifold_proof: Vec<u8>,   // serialized proof from manifold module
         pub biological_proof: Option<Vec<u8>>,
         pub combined_hash: [u8; 32],
     }
     impl AttentionProof {
         pub fn verify(&self) -> bool;
     }

4. Implement motif extraction:
   extract_motifs() identifies subgraph motifs (3-8 nodes) meeting:
     - activation_frequency >= config.min_frequency
     - predictive_power >= config.min_predictive_power
     - All attention proofs verify successfully (ADR-003 invariant)

   Motifs that fail proof verification are discarded and a ProofFailure event
   is emitted.

5. Add GnnEngine to the CognitiveGraph aggregate root:
     pub struct CognitiveGraph {
         hypergraph: HypergraphStore,   // from Prompt 08
         gnn_engine: GnnEngine,         // NEW
         healer: GraphHealer,           // from Prompt 08
         shortcuts: ShortcutRegistry,   // from Prompt 08
         config: CognitiveGraphConfig,
     }
     impl CognitiveGraph {
         pub fn discover_shortcuts(&mut self) -> Vec<CognitiveShortcut>;
     }

   discover_shortcuts():
     a) adjacency = hypergraph.to_adjacency()
     b) features = hypergraph.node_features()
     c) output = gnn_engine.forward(adjacency, features)
     d) shortcuts = gnn_engine.extract_motifs(output, config)
     e) Register each shortcut in shortcuts registry
     f) Publish ShortcutDiscovered event for each
     g) Return shortcuts

6. Define domain events:
     ShortcutDiscovered { shortcut, proof, discovery_latency_ms }
     ProofFailure { module: String, affected_subgraph: Vec<NodeId>, reason: String }

7. Update ShortcutRegistry:
     pub struct ShortcutRegistry {
         shortcuts: HashMap<ShortcutId, CognitiveShortcut>,
     }
     impl ShortcutRegistry {
         pub fn register(&mut self, shortcut: CognitiveShortcut);
         pub fn deregister(&mut self, id: &ShortcutId);
         pub fn find_matching(&self, candidates: &[SearchResult])
             -> Vec<(CognitiveShortcut, f32)>;
         pub fn all(&self) -> Vec<&CognitiveShortcut>;
     }

   find_matching() maps KNN SearchResults to their associated shortcuts.
   This is consumed by Intent Prediction (Prompt 10).

8. Write tests:
   - Build a graph with planted motifs (known subgraph patterns that repeat).
   - discover_shortcuts() finds the planted motifs.
   - AttentionProof.verify() returns true for valid proofs.
   - Motifs with failed proofs are rejected and ProofFailure is emitted.
   - ShortcutRegistry.find_matching() correctly maps search results to shortcuts.
   - Benchmark: discover_shortcuts() < 500ms for a graph with 1000 nodes.
```

### Produces

- `GnnEngine` with proof-gated attention (temporal, manifold, biological)
- `CognitiveShortcut` with verified `AttentionProof`
- `ShortcutRegistry` with `find_matching()` -- consumed by Intent Prediction (Prompt 10)
- `ShortcutDiscovered` events consumed by Provenance (Prompt 12)
- GNN layers accept LoRA deltas from Adaptation (Prompt 11)

---

## Prompt 10 -- Intent Prediction Pipeline

**Prerequisites:** Prompt 04 (VectorMemory.search_knn), Prompt 07 (TemporalLearner.get_context), Prompt 09 (ShortcutRegistry.find_matching)
**Implements:** DDD-005 (IntentPredictor aggregate, ShortcutMatcher, AttentionScorer, TemporalReweighter)
**Introduces:** `ruvector-attn-mincut` v2.0.4, `ruvector-attention-wasm` v2.0.4

### Prompt

```
Create the crate synapse-graph-prediction as a new workspace member.

Implement the IntentPredictor aggregate root per DDD-005. This is the primary
output-facing domain with the strictest latency requirement: < 1ms end-to-end.

1. Implement ShortcutMatcher:
     pub struct ShortcutMatcher {
         config: MatcherConfig,  // top_k for KNN
     }
     impl ShortcutMatcher {
         pub fn match_shortcuts(&self,
                                 partial_pattern: &GatedEmbedding,
                                 memory: &VectorMemory,          // Prompt 04
                                 registry: &ShortcutRegistry)    // Prompt 09
             -> Vec<MatchedShortcut>;
     }

   MatchedShortcut { shortcut: CognitiveShortcut, distance: f32 }.

   match_shortcuts():
     a) memory.search_knn(partial_pattern.vector(), config.top_k) -> candidates
     b) registry.find_matching(candidates) -> matched shortcuts with distances
     c) Sort by distance ascending, return top matches

2. Implement AttentionScorer wrapping ruvector-attention + ruvector-attn-mincut:
     pub struct AttentionScorer {
         attention: ruvector_attention::AttentionEngine,
         mincut_gate: ruvector_attn_mincut::MinCutGate,
     }
     impl AttentionScorer {
         pub fn score(&self,
                       query: &[f32],
                       matched: &[MatchedShortcut])
             -> Vec<ScoredIntent>;
     }

   Use min-cut gated attention (not softmax) per DDD-005:
     - query = partial_pattern vector
     - keys = shortcut centroids
     - values = intent associations
     - gate = dynamic min-cut

   Returns ScoredIntent { intent_id, confidence, shortcut_id, match_distance }.

3. Implement TemporalReweighter:
     pub struct TemporalReweighter {
         config: ReweightConfig,
     }
     impl TemporalReweighter {
         pub fn reweight(&self,
                          scored: &mut [ScoredIntent],
                          context: &TemporalContext);  // from Prompt 07
     }

   For each ScoredIntent:
     temporal_consistency = cosine_similarity(intent_pattern, context.blended())
     scored.confidence *= (1.0 + config.boost_factor * temporal_consistency)

   Intents consistent with temporal context get boosted; contradictory ones
   get penalized.

4. Implement IntentVocabulary:
     pub struct IntentVocabulary {
         intents: HashMap<IntentId, IntentEntry>,
     }
     impl IntentVocabulary {
         pub fn register(&mut self, label: String) -> IntentId;
         pub fn get(&self, id: &IntentId) -> Option<&IntentEntry>;
         pub fn record_accuracy(&mut self, id: &IntentId, correct: bool);
     }

   IntentEntry per DDD-005: { id, label, shortcut_ids, activation_count,
   accuracy_rate, first_seen, last_predicted }.

5. Assemble IntentPredictor aggregate root:
     pub struct IntentPredictor {
         matcher: ShortcutMatcher,
         scorer: AttentionScorer,
         reweighter: TemporalReweighter,
         vocabulary: IntentVocabulary,
         config: PredictionConfig,
     }
     impl IntentPredictor {
         pub fn predict(&self,
                         partial_pattern: &GatedEmbedding,
                         temporal_ctx: &TemporalContext,
                         memory: &VectorMemory,
                         registry: &ShortcutRegistry)
             -> IntentPrediction;

         pub fn record_feedback(&mut self,
                                 prediction_id: PredictionId,
                                 feedback: PredictionFeedback)
             -> Option<AdaptationSignal>;
     }

   predict() follows DDD-005 pipeline:
     a) matcher.match_shortcuts(partial_pattern, memory, registry)
     b) scorer.score(partial_pattern.vector(), matched)
     c) reweighter.reweight(scored, temporal_ctx)
     d) Sort by confidence, take top prediction_k
     e) Return IntentPrediction { id, timestamp, ranked_intents, attention_weights,
        temporal_influence, latency_us }

   record_feedback() produces an AdaptationSignal per DDD-005:
     - Confirmed -> None (no adaptation needed)
     - Corrected { actual } -> Some(AdaptationSignal { type: PredictionError,
       urgency: High })
     - Rejected -> Some(AdaptationSignal { type: PredictionError, urgency: High })
     - Timeout -> None

6. Define domain events:
     PredictionMade { prediction, feedback_window_ms }
     NewIntentDiscovered { intent_id, initial_pattern, discovery_context }
     AccuracyDegraded { current_accuracy, threshold, window_size }

7. Implement cold start per DDD-005:
   If registry has no shortcuts, predict() returns IntentPrediction with
   empty ranked_intents and publishes NewIntentDiscovered for each unique
   pattern cluster seen.

8. Write tests:
   - predict() with known shortcuts returns correct ranked intents.
   - Temporal reweighting boosts consistent predictions.
   - Cold start returns empty predictions and emits NewIntentDiscovered.
   - record_feedback(Corrected) produces AdaptationSignal with High urgency.
   - record_feedback(Confirmed) produces None.
   - AccuracyDegraded fires when rolling accuracy drops below 0.7.
   - Benchmark: full predict() pipeline < 1ms for 100 shortcuts.
```

### Produces

- `synapse-graph-prediction` crate with `IntentPredictor` aggregate
- `predict()` is the primary user-facing output of the system
- `AdaptationSignal` output consumed by Adaptation (Prompt 11)
- `PredictionMade` events consumed by Provenance (Prompt 12)
- Attention layers accept LoRA deltas from Adaptation (Prompt 11)

---

## Prompt 11 -- Two-Tier LoRA Adaptation Engine

**Prerequisites:** Prompt 09 (GnnEngine -- target for slow LoRA deltas), Prompt 10 (IntentPredictor -- source of AdaptationSignal, target for fast LoRA deltas), Prompt 07 (DriftDetected event)
**Implements:** ADR-004 (two-tier LoRA), DDD-006 (AdaptationEngine aggregate, FastLoraPath, SlowLoraPath, EWC++)
**Introduces:** `ruvector-learning-wasm` v2.0.4 (WASM LoRA bindings)

### Prompt

```
Create the crate synapse-graph-adaptation as a new workspace member.

Implement the AdaptationEngine aggregate root per DDD-006 and ADR-004.
ruvector-sona was already introduced in Prompt 07 for slow-tier consolidation;
this prompt builds the full two-tier LoRA adaptation system on top of it.

1. Define LoraDelta per DDD-006:
     pub struct LoraDelta {
         pub layer_id: LayerId,
         pub matrix_a: Vec<f32>,  // shape: (d_out, rank)
         pub matrix_b: Vec<f32>,  // shape: (d_in, rank)
         pub rank: usize,
         pub computed_at: Timestamp,
         pub trigger: AdaptationType,
     }
     impl LoraDelta {
         pub fn memory_bytes(&self) -> usize;
         pub fn summary(&self) -> DeltaSummary;
     }

2. Implement FastLoraPath:
     pub struct FastLoraPath {
         rank: usize,  // 4
         target_layers: Vec<LayerId>,  // attention heads + classification
         accumulated_deltas: Vec<LoraDelta>,
         damping_factor: f32,
         max_accumulated: usize,
     }
     impl FastLoraPath {
         pub fn compute_delta(&mut self, signal: &AdaptationSignal,
                               sona: &mut SonaEngine) -> LoraDelta;
     }

   compute_delta():
     a) Compute error gradient from signal (prediction vs actual).
     b) Use sona.fast_adapt(gradient, rank=4) to get rank-4 LoRA matrices.
     c) Apply exponential moving average with damping_factor to prevent oscillation.
     d) Append to accumulated_deltas.
     e) Return the delta.

   Latency target: < 100 microseconds.

3. Implement SlowLoraPath:
     pub struct SlowLoraPath {
         rank: usize,  // 16
         target_layers: Vec<LayerId>,  // all layers
         ewc_lambda: f32,
         persisted_deltas: Vec<LoraDelta>,
         last_consolidation: Timestamp,
     }
     impl SlowLoraPath {
         pub fn consolidate(&mut self, accumulated: Vec<LoraDelta>,
                             fisher: &FisherInformationMatrix,
                             sona: &mut SonaEngine) -> ConsolidationResult;
     }

   consolidate():
     a) Aggregate accumulated fast deltas.
     b) For each target layer:
        - Compute rank-16 LoRA delta via sona.slow_adapt(gradient, rank=16).
        - Apply EWC++ penalty: loss += lambda * sum(F_i * (theta_i - theta_i*)^2).
        - Clip delta if EWC penalty exceeds threshold.
     c) Persist deltas to storage.
     d) Return ConsolidationResult { deltas_consolidated, total_latency,
        ewc_penalty, weights_protected, fisher_recomputed }.

   Latency target: < 10 milliseconds.

4. Implement FisherInformationMatrix:
     pub struct FisherInformationMatrix {
         diagonal: Vec<f32>,
         computed_at: Timestamp,
         sample_count: u64,
     }
     impl FisherInformationMatrix {
         pub fn compute(graph: &CognitiveGraph, sona: &SonaEngine)
             -> FisherInformationMatrix;
         pub fn importance(&self, weight_index: usize) -> f32;
     }

   Uses ruvector-math for the diagonal Fisher approximation.

5. Implement OscillationDetector:
     pub struct OscillationDetector {
         sign_history: HashMap<LayerId, Vec<bool>>,  // true = positive, false = negative
         window_size: usize,
         flip_threshold: usize,
     }
     impl OscillationDetector {
         pub fn record(&mut self, layer_id: LayerId, delta: &LoraDelta);
         pub fn is_oscillating(&self, layer_id: &LayerId) -> bool;
         pub fn damping_multiplier(&self, layer_id: &LayerId) -> f32;
     }

6. Assemble AdaptationEngine aggregate root:
     pub struct AdaptationEngine {
         sona: SonaEngine,                // wraps ruvector-sona
         fast_path: FastLoraPath,
         slow_path: SlowLoraPath,
         fisher: FisherInformationMatrix,
         oscillation: OscillationDetector,
         config: AdaptationConfig,
     }
     impl AdaptationEngine {
         pub fn adapt(&mut self, signal: AdaptationSignal) -> AdaptationResult;
         pub fn consolidate(&mut self) -> ConsolidationResult;
         pub fn last_delta(&self) -> Option<&LoraDelta>;
         pub fn last_latency(&self) -> Duration;
         pub fn recompute_fisher(&mut self, graph: &CognitiveGraph);
     }

   adapt():
     a) Match signal.urgency:
        High -> fast_path.compute_delta(signal, sona)
        Low  -> queue signal for next consolidate() call
     b) Check oscillation detector; apply damping if oscillating.
     c) Return AdaptationResult { delta, latency, path, ewc_penalty }.

   consolidate():
     a) Drain fast_path.accumulated_deltas.
     b) slow_path.consolidate(accumulated, fisher, sona).
     c) Optionally recompute Fisher if stale.
     d) Return ConsolidationResult.

7. Define the delta propagation trait (consumed by GNN and Attention):
     pub trait LoraTarget {
         fn apply_lora_delta(&mut self, delta: &LoraDelta) -> Result<(), LoraError>;
         fn layer_ids(&self) -> Vec<LayerId>;
     }

   Add LoraTarget implementations to:
     - GnnEngine (Prompt 09) -- for slow LoRA deltas
     - AttentionScorer (Prompt 10) -- for fast LoRA deltas

8. Define domain events:
     AdaptationApplied { result, affected_layers, trigger }
     Consolidated { result, accumulated_fast_deltas_count }
     OscillationDetected { layer_id, magnitude, damping_applied }
     FisherRecomputed { sample_count, top_protected_weights }

9. Write tests:
   - adapt() with High urgency signal produces fast LoRA delta in < 100us.
   - adapt() with Low urgency signal queues for consolidation.
   - consolidate() applies EWC++ regularization (protected weights change less).
   - Oscillation detector detects sign-flip pattern and reduces learning rate.
   - LoRA deltas successfully applied to GnnEngine and AttentionScorer.
   - Persisted deltas survive engine reconstruction (serialize/deserialize).
   - Benchmark: fast adapt < 100us, consolidate < 10ms.
```

### Produces

- `synapse-graph-adaptation` crate with `AdaptationEngine` aggregate
- `LoraTarget` trait implemented on GnnEngine (Prompt 09) and AttentionScorer (Prompt 10)
- `AdaptationApplied`/`Consolidated` events consumed by Provenance (Prompt 12)
- Completes the prediction-feedback-adaptation loop

---

## Prompt 12 -- Provenance DAG & Event Logging

**Prerequisites:** Prompt 03 (ingestion events), Prompt 04 (storage events), Prompt 06 (promotion events), Prompt 07 (drift events), Prompt 08 (healing events), Prompt 09 (discovery events), Prompt 10 (prediction events), Prompt 11 (adaptation events)
**Implements:** DDD-007 (ProvenanceLog aggregate, DagStore, ProvenanceQueryEngine, NeuralEvent)
**Introduces:** `ruvector-dag` v2.0.4

### Prompt

```
Create the crate synapse-graph-provenance as a new workspace member.

Implement the ProvenanceLog aggregate root per DDD-007. This is the terminal
event sink -- it consumes domain events from ALL other bounded contexts.

1. Define NeuralEvent as a union enum per DDD-007:
     pub enum NeuralEvent {
         // From Spike Ingestion (DDD-001, Prompt 03)
         Ingested(EmbeddingIngested),
         Denied(EmbeddingDenied),
         ChannelHealth(ChannelHealthChanged),

         // From Vector Memory (DDD-002, Prompt 04)
         Stored(EmbeddingStored),
         Evicted(EmbeddingsEvicted),

         // From Temporal Learning (DDD-003, Prompts 06-07)
         Promoted(PatternPromoted),
         TierEvicted(TierEvicted),
         Drift(DriftDetected),

         // From Cognitive Graph (DDD-004, Prompts 08-09)
         ShortcutFound(ShortcutDiscovered),
         Healed(GraphHealed),
         ProofFail(ProofFailure),
         TopologyChanged(GraphTopologyChanged),

         // From Intent Prediction (DDD-005, Prompt 10)
         Predicted(PredictionMade),
         NewIntent(NewIntentDiscovered),
         AccuracyDrop(AccuracyDegraded),

         // From Adaptation (DDD-006, Prompt 11)
         Adapted(AdaptationApplied),
         Consolidated(Consolidated),
         Oscillation(OscillationDetected),
         FisherUpdated(FisherRecomputed),
     }

   Place this in synapse-graph-types so all crates can construct their variants.
   NeuralEvent must derive Serialize, Deserialize.

2. Implement EventNode:
     pub struct EventNode {
         pub id: EventId,
         pub timestamp: Timestamp,
         pub session_id: SessionId,
         pub event_type: EventType,
         pub payload: Vec<u8>,             // serialized NeuralEvent
         pub predecessor_ids: Vec<EventId>,
         pub hash: [u8; 32],               // SHA-256(payload + predecessor hashes)
     }

   EventType enum: Ingestion, Denial, Storage, Eviction, TierPromotion,
   DriftDetection, ShortcutDiscovery, GraphHealing, ProofFailure, Prediction,
   Adaptation, Consolidation, SystemHealth.

3. Implement DagStore wrapping ruvector-dag:
     pub struct DagStore {
         inner: ruvector_dag::Dag,
         next_id: EventId,
     }
     impl DagStore {
         pub fn append(&mut self, event: NeuralEvent, session: SessionId,
                        predecessors: Vec<EventId>) -> EventId;
         pub fn get(&self, id: EventId) -> Option<&EventNode>;
         pub fn total_events(&self) -> u64;
     }

   append():
     a) Serialize event to payload bytes.
     b) Compute hash: SHA-256(payload || predecessor_0.hash || ... || predecessor_n.hash).
     c) Create EventNode with monotonically increasing ID.
     d) Store in ruvector-dag.
     e) Return EventId.

   Performance: < 5 microseconds per append.

4. Implement ProvenanceQueryEngine:
     pub struct ProvenanceQueryEngine<'a> {
         dag: &'a DagStore,
     }
     impl<'a> ProvenanceQueryEngine<'a> {
         pub fn trace(&self, event_id: EventId) -> AuditTrail;
         pub fn session_slice(&self, session_id: SessionId) -> Vec<&EventNode>;
         pub fn shortcut_origin(&self, shortcut_id: ShortcutId) -> Option<&EventNode>;
         pub fn adaptation_history(&self, layer_id: LayerId) -> Vec<&EventNode>;
         pub fn verify_integrity(&self) -> IntegrityReport;
     }

   trace() walks predecessor_ids backward to the root events, building the
   complete causal chain.

   verify_integrity() recomputes every hash and checks for mismatches.

   AuditTrail { target_event, causal_chain, depth, domains_involved }.
   IntegrityReport { total_events, verified_events, corrupted_events, is_valid }.

5. Implement causal predecessor resolution per DDD-007:

   Define a CausalResolver that, given a NeuralEvent and the current DAG state,
   determines the predecessor EventIds:

     EmbeddingIngested -> predecessors: [] (root event)
     EmbeddingStored   -> predecessors: [corresponding EmbeddingIngested event]
     EmbeddingDenied   -> predecessors: [] (root event)
     PatternPromoted   -> predecessors: [EmbeddingIngested events in the pattern]
     ShortcutFound     -> predecessors: [PatternPromoted + EmbeddingStored events]
     Predicted         -> predecessors: [ShortcutFound events for matched shortcuts]
     Adapted           -> predecessors: [the Predicted event that triggered it]
     Consolidated      -> predecessors: [all Adapted events being consolidated]
     Healed            -> predecessors: [the TopologyChanged event that triggered it]

6. Assemble ProvenanceLog aggregate root:
     pub struct ProvenanceLog {
         dag: DagStore,
         resolver: CausalResolver,
         current_session: SessionId,
     }
     impl ProvenanceLog {
         pub fn append(&mut self, event: NeuralEvent) -> EventId;
         pub fn query(&self) -> ProvenanceQueryEngine;
         pub fn verify(&self) -> IntegrityReport;
         pub fn set_session(&mut self, session: SessionId);
     }

   append() uses the CausalResolver to determine predecessors automatically,
   then delegates to dag.append().

7. Write tests:
   - Append 100 events of mixed types; verify all are retrievable.
   - trace() on a Predicted event returns the causal chain back through
     ShortcutFound -> PatternPromoted -> EmbeddingIngested.
   - session_slice() returns only events from the specified session.
   - verify_integrity() passes on a clean DAG.
   - Tamper with one event's payload; verify_integrity() detects the corruption.
   - Benchmark: append() < 5us, trace() depth=10 < 100us.
```

### Produces

- `synapse-graph-provenance` crate with `ProvenanceLog` aggregate
- Complete `NeuralEvent` union enum in shared types (all event types from all domains)
- `CausalResolver` automatically links events into a tamper-evident DAG
- Query engine for audit trails, session slices, and integrity verification

---

## Prompt 13 -- Application Orchestrator (Wiring All Domains Together)

**Prerequisites:** ALL prior prompts (01-12)
**Implements:** SPARC pseudocode (initialize_synapse_graph, ingest_spike_train, full data flow), DDD context map (all inter-domain relationships)
**Introduces:** No new crates (uses all previously introduced crates)

### Prompt

```
Implement the application orchestrator in synapse-graph-core that wires all
bounded contexts together per the SPARC pseudocode and the DDD context map
in INDEX.md.

1. Define the top-level SynapseGraph struct:
     pub struct SynapseGraph {
         ingester: SpikeIngester,           // Prompt 03
         memory: VectorMemory,              // Prompt 04
         temporal: TemporalLearner,         // Prompt 07
         cognitive: CognitiveGraph,         // Prompts 08-09
         predictor: IntentPredictor,        // Prompt 10
         adaptation: AdaptationEngine,      // Prompt 11
         provenance: ProvenanceLog,         // Prompt 12
         config: SynapseGraphConfig,
     }

2. Implement initialization per SPARC pseudocode initialize_synapse_graph():
     pub fn new(config: SynapseGraphConfig) -> Self;

   Initialize each subsystem with appropriate config sections.

3. Implement the main ingestion pipeline per SPARC ingest_spike_train():
     pub fn ingest(&mut self, batch: SpikeBatch) -> IngestResult {
         // a) Spike processing + coherence gating
         let result = self.ingester.ingest(batch);

         // b) Log denied embeddings to provenance
         for denied in &result.denied {
             self.provenance.append(NeuralEvent::Denied(denied.clone()));
         }

         // c) For each gated embedding:
         for gated in &result.gated {
             // c1) Classify temporal tier
             let timing = result.timing_for(&gated);
             let tier = self.temporal.router().classify(gated, &timing);
             let meta = TemporalMeta { tier, timestamp: gated.timestamp(),
                                        channel_id: gated.channel_id(),
                                        session_id: gated.session_id(),
                                        coherence_score: gated.coherence_score() };

             // c2) Store in vector memory
             let id = self.memory.insert(gated.clone(), meta.clone())?;

             // c3) Accumulate in temporal learner
             self.temporal.accumulate(gated, &timing);

             // c4) Log ingestion to provenance
             self.provenance.append(NeuralEvent::Ingested(
                 EmbeddingIngested { embedding_id: id, ... }));
         }

         // d) Log channel health changes
         for change in &result.channel_health_changes {
             self.provenance.append(NeuralEvent::ChannelHealth(change.clone()));
         }

         Ok(IngestResult { ingested: result.gated.len(), denied: result.denied.len() })
     }

4. Implement the cognitive graph update cycle:
     pub fn update_cognitive_graph(&mut self) -> GraphUpdateResult {
         // a) Cluster recent embeddings from vector memory
         let clusters = self.memory.cluster_recent(self.config.cluster_threshold);

         // b) Extract temporal sequences from temporal learner
         let sequences: Vec<TemporalSequence> = clusters.iter()
             .flat_map(|c| self.temporal.extract_sequences(c))
             .collect();

         // c) Find temporally adjacent pairs for co-activation edges
         let neighbors: Vec<TemporalNeighbor> = clusters.iter()
             .flat_map(|c| self.memory.find_temporally_adjacent(c, Duration::from_millis(10)))
             .collect();

         // d) Update the cognitive graph
         let result = self.cognitive.update(clusters, sequences, neighbors);

         // e) Log to provenance
         self.provenance.append(NeuralEvent::TopologyChanged(...));

         result
     }

5. Implement shortcut discovery:
     pub fn discover_shortcuts(&mut self) -> Vec<CognitiveShortcut> {
         let shortcuts = self.cognitive.discover_shortcuts();
         for s in &shortcuts {
             self.provenance.append(NeuralEvent::ShortcutFound(
                 ShortcutDiscovered { shortcut: s.clone(), ... }));
         }
         shortcuts
     }

6. Implement intent prediction:
     pub fn predict_intent(&self, partial: &GatedEmbedding) -> IntentPrediction {
         let ctx = self.temporal.get_context(self.config.tier_weights);
         let prediction = self.predictor.predict(
             partial, &ctx, &self.memory, self.cognitive.shortcuts());
         // provenance logging of PredictionMade happens here
         prediction
     }

7. Implement the feedback -> adaptation loop:
     pub fn provide_feedback(&mut self, prediction_id: PredictionId,
                              feedback: PredictionFeedback) {
         if let Some(signal) = self.predictor.record_feedback(prediction_id, feedback) {
             let result = self.adaptation.adapt(signal);

             // Propagate LoRA deltas to GNN and attention
             if let Some(delta) = self.adaptation.last_delta() {
                 self.cognitive.gnn_engine_mut().apply_lora_delta(delta);
                 self.predictor.scorer_mut().apply_lora_delta(delta);
             }

             self.provenance.append(NeuralEvent::Adapted(
                 AdaptationApplied { result, ... }));
         }
     }

8. Implement maintenance cycle:
     pub fn maintain(&mut self) {
         // a) Temporal tier maintenance (promotion/eviction)
         let temporal_result = self.temporal.maintain();
         for p in temporal_result.promoted {
             self.provenance.append(NeuralEvent::Promoted(p));
         }

         // b) Graph self-healing
         let healing = self.cognitive.heal();
         self.provenance.append(NeuralEvent::Healed(healing));

         // c) Slow LoRA consolidation (if idle)
         let consolidation = self.adaptation.consolidate();
         self.provenance.append(NeuralEvent::Consolidated(consolidation));
     }

9. Implement session lifecycle:
     pub fn start_session(&mut self, session_id: SessionId);
     pub fn end_session(&mut self);
       // end_session calls self.temporal.end_session() and
       // triggers a final consolidation pass.

10. Write integration tests:
    - Full pipeline: MockBciAdapter -> ingest() -> update_cognitive_graph()
      -> discover_shortcuts() -> predict_intent() -> provide_feedback()
    - Verify provenance DAG contains the complete causal chain.
    - Session lifecycle: start_session, ingest 1000 batches, end_session,
      verify temporal tier promotions occurred.
    - Feedback loop: inject prediction error, verify LoRA adaptation fires,
      verify GNN and attention weights are updated.
    - Maintenance cycle: verify graph healing and consolidation run.
    - Benchmark: full ingest cycle (ingest + temporal + provenance) < 5ms
      for a 1024-channel batch.
```

### Produces

- `synapse-graph-core` with the `SynapseGraph` application orchestrator
- All inter-domain data flows wired per the DDD context map
- Complete prediction-feedback-adaptation loop
- Session lifecycle management
- Maintenance cycle (temporal promotion, graph healing, LoRA consolidation)

---

## Prompt 14 -- WASM Compilation & Edge Deployment

**Prerequisites:** Prompt 13 (complete application orchestrator)
**Implements:** ADR-006 (WASM-first deployment)
**Introduces:** `ruvector-gnn-wasm` v2.0.4, `ruvector-attention-wasm` v2.0.4, `ruvector-learning-wasm` v2.0.4, `ruvector-tiny-dancer-wasm` v2.0.4

### Prompt

```
Implement the WASM compilation target in synapse-graph-wasm per ADR-006.

1. Configure synapse-graph-wasm Cargo.toml for wasm32-wasi target:
   - Set crate-type = ["cdylib"]
   - Enable wasm-bindgen or wasm32-wasi as appropriate
   - Import synapse-graph-core and all domain crates
   - Enable the WASM variants of ruvector crates:
       ruvector-wasm, ruvector-gnn-wasm, ruvector-attention-wasm,
       ruvector-nervous-system-wasm, ruvector-learning-wasm,
       ruvector-tiny-dancer-wasm

2. Create a WASM-compatible initialization function:
     #[wasm_bindgen]  // or WASI export
     pub fn init_synapse_graph(config_json: &str) -> *mut SynapseGraph;

3. Create WASM-exported functions for each operation:
     pub fn wasm_ingest(sg: *mut SynapseGraph, batch_bytes: &[u8]) -> Vec<u8>;
     pub fn wasm_predict(sg: *mut SynapseGraph, pattern_bytes: &[u8]) -> Vec<u8>;
     pub fn wasm_feedback(sg: *mut SynapseGraph, prediction_id: u64,
                           feedback_type: u8);
     pub fn wasm_maintain(sg: *mut SynapseGraph);
     pub fn wasm_discover_shortcuts(sg: *mut SynapseGraph) -> Vec<u8>;
     pub fn wasm_start_session(sg: *mut SynapseGraph, session_id: u64);
     pub fn wasm_end_session(sg: *mut SynapseGraph);

   Use a binary serialization format (e.g., bincode or the RVF format from
   @ruvector/rvf-wasm) for input/output to minimize serialization overhead.

4. Implement arena allocators for the fast-tier ring buffer to avoid WASM
   linear memory fragmentation (per ADR-006 risk mitigation).

5. Ensure all modules select WASM-compatible paths:
   - CoherenceGate uses cognitum-gate-kernel (already no_std)
   - NervousSystem uses ruvector-nervous-system-wasm
   - GNN uses ruvector-gnn-wasm
   - Attention uses ruvector-attention-wasm
   - Learning uses ruvector-learning-wasm
   - Router uses ruvector-tiny-dancer-wasm

6. Build and verify:
   - cargo build --target wasm32-wasi -p synapse-graph-wasm
   - Total WASM binary size < 4 MB (per ADR-006)
   - Run the integration test from Prompt 13 against the WASM module
     (using wasmtime or wasmer as the host runtime)

7. Implement the JavaScript bridge using the npm packages:
   Create a TypeScript file src/js/synapse-graph.ts that:
     a) Loads the WASM module using @ruvector/rvf-wasm
     b) Provides a typed TypeScript API wrapping the WASM exports
     c) Uses ruvector (npm) for any vector operations needed in the JS layer
     d) Uses @ruvector/gnn (npm) for GNN operations accessible from Node.js

8. Write tests:
   - WASM module loads successfully in wasmtime.
   - Full pipeline runs through WASM exports (ingest -> predict -> feedback).
   - Benchmark: WASM latencies are within 3x of native for all operations.
   - Memory: WASM linear memory usage < 64 MB under sustained load.
   - Binary size: total < 4 MB.
```

### Produces

- `synapse-graph-wasm` crate compiled to wasm32-wasi
- WASM exports for all major operations
- TypeScript bridge in `src/js/synapse-graph.ts`
- Verification that all latency and memory budgets are met in WASM

---

## Prompt 15 -- End-to-End Integration, Benchmarking & Hardening

**Prerequisites:** Prompt 14 (WASM target working)
**Implements:** SPARC Completion Phase 5, Verification & Testing Strategy, Success Criteria
**Introduces:** `ruvector-sparse-inference` v0.1.31, `ruv-fann` v0.2.0 (native fallback), optional: `ruvnet/Synaptic-Mesh`, `ruvnet/QuDAG`

### Prompt

```
Perform end-to-end integration testing, benchmarking, and production hardening
per SPARC Phase 5 and the Success Criteria.

1. Build a comprehensive BCI simulation harness:
   Extend MockBciAdapter to generate realistic multi-session synthetic data:
     a) Session 1-3: establish baseline patterns (5 distinct intents).
     b) Session 4-6: introduce a new intent pattern (tests cold-start -> learning).
     c) Session 7-10: simulate gradual neural drift (1% per session).
     d) Session 11: simulate electrode failure on 10% of channels.
     e) Session 12-15: recovery after electrode failure.

   Each session produces 10,000 spike batches of 1024 channels.

2. Run the full pipeline across all 15 simulated sessions:
     for session in 1..=15 {
         sg.start_session(session);
         for batch in simulator.generate_session(session) {
             sg.ingest(batch);
         }
         // Periodic cognitive graph update
         sg.update_cognitive_graph();
         sg.discover_shortcuts();
         // Run predictions on held-out patterns
         for test_pattern in simulator.test_patterns(session) {
             let prediction = sg.predict_intent(&test_pattern);
             let feedback = simulator.evaluate(prediction, test_pattern);
             sg.provide_feedback(prediction.id, feedback);
         }
         sg.maintain();
         sg.end_session();
     }

3. Verify Success Criteria:
   a) FUNCTIONAL: prediction accuracy > 80% by session 6 for established intents.
   b) LATENCY: measure and assert:
      - End-to-end ingestion-to-prediction < 1 ms
      - LoRA adaptation < 100 us
      - Coherence gate < 1 us
   c) MEMORY: runtime RSS < 64 MB on WASM target.
   d) ADAPTATION: accuracy improves by > 20% between sessions 1 and 15 without
      catastrophic forgetting of session 1-3 patterns.
   e) PROVENANCE: trace a session-15 prediction back to its root ingestion events.
      Verify the causal chain is complete and hashes are valid.

4. Latency benchmark suite (using Criterion.rs):
   Create benchmarks for every operation in the performance budget table:
     - Spike ingestion + BTSP: < 100us
     - Coherence gate: < 1us
     - Temporal routing: < 10us
     - HNSW insert: < 50us
     - HNSW KNN (k=10): < 200us
     - GNN shortcut discovery: < 500ms
     - Intent prediction: < 1ms
     - Fast LoRA: < 100us
     - Slow LoRA: < 10ms
     - Min-cut analysis: < 100ms
     - DAG append: < 5us

5. Adversarial testing:
   a) Fuzz the coherence gate with random embeddings; verify 0% leakage of
      out-of-distribution inputs.
   b) Inject adversarial embeddings designed to create false shortcuts;
      verify proof-gated attention rejects them.
   c) Simulate rapid conflicting predictions; verify oscillation detector
      activates and dampens.

6. Provenance integrity:
   After the full 15-session simulation, call provenance.verify() and assert
   IntegrityReport.is_valid == true with 0 corrupted events.

7. Integrate native-only optimizations:
   a) ruv-fann v0.2.0 as the CPU-native neural engine for non-WASM targets.
   b) ruvector-sparse-inference v0.1.31 for PowerInfer-style sparse execution.
   Both should be feature-gated: enabled for native builds, disabled for WASM.

8. Optional distributed integration:
   If time permits, integrate ruvnet/Synaptic-Mesh + ruvnet/QuDAG for
   multi-device provenance replication. This extends DDD-007 with distributed
   DAG consensus but is NOT required for the core success criteria.

9. Generate a test report documenting:
   - Per-session accuracy curves
   - Latency percentiles (p50, p95, p99) for each operation
   - Memory usage over time
   - Number of shortcuts discovered per session
   - Promotion/eviction counts per session
   - Adaptation event counts and EWC penalty trends
   - Provenance DAG statistics (total events, max causal depth)
```

### Produces

- Complete end-to-end integration test covering 15 simulated BCI sessions
- Criterion.rs benchmark suite for all critical-path operations
- Adversarial test suite for coherence gate and proof-gated attention
- Provenance integrity verification
- Native-only performance optimizations (feature-gated)
- Test report documenting all success criteria metrics

---

## Dependency Graph Summary

```
Prompt 01 (Shared Types)
  |
  +-- Prompt 02 (CoherenceGate + GatedEmbedding)
  |     |
  |     +-- Prompt 03 (SpikeIngester + NervousSystem + BCI Adapter)
  |     |
  |     +-- Prompt 04 (VectorMemory + HNSW + Collections)
  |     |     |
  |     |     +-- Prompt 08 (Cognitive Graph + Hypergraph + Healer)
  |     |           |
  |     |           +-- Prompt 09 (GNN Engine + Shortcut Discovery)
  |     |                 |
  |     |                 +-- Prompt 10 (Intent Prediction)
  |     |                 |     |
  |     |                 |     +-- Prompt 11 (Adaptation Engine)
  |     |                 |           |
  |     |                 |           +-- Prompt 12 (Provenance DAG)
  |     |                 |                 |
  |     |                 |                 +-- Prompt 13 (Orchestrator)
  |     |                 |                       |
  |     |                 |                       +-- Prompt 14 (WASM)
  |     |                 |                             |
  |     |                 |                             +-- Prompt 15 (Integration)
  |     |                 |
  |     +-- Prompt 05 (Temporal Router + Fast Tier)
  |           |
  |           +-- Prompt 06 (Medium Tier + Promotion)
  |                 |
  |                 +-- Prompt 07 (Slow Tier + TemporalLearner Aggregate)
  |                       |
  |                       +-- (feeds into Prompt 08 via TemporalSequence)
  |                       +-- (feeds into Prompt 10 via TemporalContext)
  |                       +-- (feeds into Prompt 11 via DriftDetected)
```

### Parallel Execution Opportunities

While the prompts must be executed sequentially for correctness, the following
pairs can be developed in parallel if multiple developers are available:

- **Prompt 03 + Prompt 04**: SpikeIngester and VectorMemory are independent
  (both depend on Prompt 02 but not on each other).
- **Prompt 05 + Prompt 04**: TemporalRouter and VectorMemory are independent
  (both depend on Prompt 02).
- **Prompt 08 + Prompt 07**: Cognitive Graph and TemporalLearner can be
  partially parallelized. Prompt 08 needs TemporalSequence (from Prompt 06)
  but not the complete TemporalLearner aggregate (Prompt 07).

### Crate Introduction Order

| Prompt | Crates Introduced |
|--------|-------------------|
| 01 | `ruvector-core`, `ruvector-math`, `ruvector-collections` |
| 02 | `cognitum-gate-kernel` |
| 03 | `ruvector-nervous-system`, `ruvector-nervous-system-wasm` |
| 04 | `ruvector-wasm` |
| 05 | `ruvector-router-core`, `ruvector-tiny-dancer-core`, `ruvector-tiny-dancer-wasm` |
| 06 | `ruvector-temporal-tensor` |
| 07 | `ruvector-sona` |
| 08 | `ruvector-graph`, `ruvector-mincut` |
| 09 | `ruvector-gnn`, `ruvector-graph-transformer`, `ruvector-gnn-wasm`, `ruvector-attention` |
| 10 | `ruvector-attn-mincut`, `ruvector-attention-wasm` |
| 11 | `ruvector-learning-wasm` |
| 12 | `ruvector-dag` |
| 13 | (none -- wiring only) |
| 14 | (WASM bindings already introduced; JavaScript bridge) |
| 15 | `ruvector-sparse-inference`, `ruv-fann` |
