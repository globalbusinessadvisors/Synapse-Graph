# DDD-003: Temporal Learning Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R5 | ADR-001
**Implementing Phase:** Phase 2

---

## Domain Overview

The Temporal Learning domain implements the three-tier temporal architecture defined in ADR-001. It owns the classification of embeddings into temporal tiers (fast, medium, slow), the accumulation and compression of temporal data at each tier, and the promotion/eviction logic that moves patterns between tiers as they persist or decay.

This domain is the bridge between raw embedding storage (Vector Memory) and higher-level pattern analysis (Cognitive Graph, Intent Prediction).

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Temporal Tier** | One of three timescale buckets: Fast (< 10 ms), Medium (minutes-hours), Slow (days-months). |
| **Temporal Router** | The classifier that assigns each embedding to exactly one temporal tier based on its timing characteristics. Uses FastGRNN for low-latency classification. |
| **FastGRNN** | Fast, Accurate, Stable, and Tiny Gated Recurrent Neural Network. A compact RNN architecture used for temporal routing. |
| **Ring Buffer** | The fast tier's data structure. Fixed-size circular buffer of raw embeddings. Oldest entries are evicted on overflow. |
| **Session Tensor** | The medium tier's data structure. A compressed tensor representation of embeddings accumulated over a session, with 8-bit quantization. |
| **Drift Vector** | The slow tier's data structure. A heavily quantized (4-bit) representation of long-term neural encoding drift. |
| **Tier Promotion** | Moving a pattern from a faster tier to a slower tier when it demonstrates persistence. Fast -> Medium: pattern persists > 500 ms. Medium -> Slow: pattern stable across 3+ sessions. |
| **Tier Eviction** | Removing data from a tier due to age or memory pressure. Fast: ring buffer overflow. Medium: session end + compression. Slow: further quantization (never full eviction). |
| **Temporal Context** | A blended representation across all three tiers, produced by `get_context()` with configurable tier weights. Used by Intent Prediction for temporal-aware scoring. |
| **Temporal Sequence** | An ordered pair of embedding clusters observed in a consistent temporal order across multiple occurrences. Exported to the Cognitive Graph as edges. |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 TEMPORAL LEARNING CONTEXT                          |
|                                                                   |
|  Inbound:                                                         |
|    EmbeddingIngested event (from Spike Ingestion)                 |
|    GatedEmbedding + TemporalMeta (from Vector Memory)             |
|                                                                   |
|  +---------------------+                                         |
|  | TemporalLearner     |  Aggregate Root                         |
|  | (owns router +      |  Orchestrates tier classification,      |
|  |  all three tiers)   |  accumulation, promotion, eviction      |
|  +----------+----------+                                         |
|             |                                                     |
|    +--------+--------+---------+                                  |
|    |                 |         |                                   |
|    v                 v         v                                   |
|  +----------+  +----------+ +----------+                          |
|  | FastTier |  | MedTier  | | SlowTier |                          |
|  | (ring    |  | (session | | (drift   |                          |
|  |  buffer) |  |  tensor) | |  vector) |                          |
|  +----------+  +----------+ +----------+                          |
|                                                                   |
|  +---------------------+                                         |
|  | TemporalRouter      |  Domain Service                         |
|  | (FastGRNN classifier)|                                        |
|  +---------------------+                                         |
|                                                                   |
|  Outbound:                                                        |
|    TemporalContext --> [Intent Prediction Context]                 |
|    TemporalSequence --> [Cognitive Graph Context]                  |
|    TierPromoted / TierEvicted --> [Provenance Context]             |
|                                                                   |
+------------------------------------------------------------------+
```

### Context Map

| Relationship | Upstream | Downstream | Type |
|-------------|----------|------------|------|
| Spike Ingestion -> Temporal Learning | Spike Ingestion (DDD-001) | Temporal Learning | Conformist (consumes EmbeddingIngested events) |
| Temporal Learning -> Cognitive Graph | Temporal Learning | Cognitive Graph (DDD-003a) | Published Language (TemporalSequence) |
| Temporal Learning -> Intent Prediction | Temporal Learning | Intent Prediction (DDD-004) | Open Host Service (TemporalContext API) |
| Temporal Learning -> Adaptation | Temporal Learning | Adaptation (DDD-005) | Published Language (DriftDetected event) |
| Temporal Learning -> Provenance | Temporal Learning | Provenance (DDD-006) | Published Language (tier events) |

---

## Domain Model

### Aggregates

#### TemporalLearner (Aggregate Root)

```rust
/// Aggregate root for the three-tier temporal learning system.
/// Invariant: every embedding is classified into exactly one tier.
/// Invariant: promotion and eviction maintain memory budget constraints.
struct TemporalLearner {
    router: TemporalRouter,
    fast_tier: FastTier,
    medium_tier: MediumTier,
    slow_tier: SlowTier,
    config: TemporalConfig,
}

impl TemporalLearner {
    /// Classify and accumulate an embedding into the appropriate tier.
    fn accumulate(&mut self, embedding: &GatedEmbedding, meta: &TemporalMeta);

    /// Get blended temporal context across all tiers.
    fn get_context(&self, weights: TierWeights) -> TemporalContext;

    /// Extract temporal sequences from the medium tier for cognitive graph construction.
    fn extract_sequences(&self, cluster: &EmbeddingCluster) -> Vec<TemporalSequence>;

    /// Run promotion and eviction logic. Called periodically.
    fn maintain(&mut self) -> MaintenanceResult;
}
```

### Entities

#### FastTier

```rust
/// Fast tier: ring buffer of raw embeddings for < 10ms patterns.
/// Identity: singleton within TemporalLearner.
struct FastTier {
    buffer: RingBuffer<TimestampedEmbedding>,  // fixed capacity
    capacity: usize,
    write_head: usize,
    pattern_detector: BurstPatternDetector,    // detects persistent patterns for promotion
}
```

#### MediumTier

```rust
/// Medium tier: session-level compressed tensors (minutes-hours).
/// Identity: singleton within TemporalLearner.
struct MediumTier {
    current_session: SessionTensor,            // ruvector-temporal-tensor
    past_sessions: Vec<CompressedSessionTensor>,
    quantization_bits: u8,                     // default: 8
    max_sessions: usize,                       // how many past sessions to retain
}
```

#### SlowTier

```rust
/// Slow tier: long-term drift vectors (days-months).
/// Identity: singleton within TemporalLearner.
struct SlowTier {
    drift_vectors: Vec<DriftVector>,           // ruvector-sona
    quantization_bits: u8,                     // default: 4
    consolidation_engine: SonaConsolidator,    // EWC++ consolidation
    last_consolidation: Timestamp,
}
```

### Value Objects

#### TemporalTier (Enum)

```rust
enum TemporalTier {
    Fast,    // < 10 ms patterns (spike bursts)
    Medium,  // Minutes to hours (session patterns)
    Slow,    // Days to months (long-term drift)
}
```

#### TemporalContext

```rust
/// Blended temporal context across all three tiers.
/// Used by intent prediction to weight predictions by recency and persistence.
struct TemporalContext {
    fast_component: Vec<f32>,     // recent spike burst context
    medium_component: Vec<f32>,   // current session context
    slow_component: Vec<f32>,     // long-term baseline
    weights: TierWeights,         // fast=0.5, medium=0.3, slow=0.2
}

impl TemporalContext {
    fn blended(&self) -> Vec<f32>;  // weighted combination
}
```

#### TemporalSequence

```rust
/// An ordered pattern observed across temporal tiers.
/// Exported to the Cognitive Graph as a TEMPORAL_SEQUENCE edge.
struct TemporalSequence {
    from_cluster: ClusterId,
    to_cluster: ClusterId,
    confidence: f32,          // frequency * consistency
    observed_count: u32,
    average_delay: Duration,
}
```

#### TierWeights

```rust
/// Configurable weights for blending temporal tiers.
struct TierWeights {
    fast: f32,     // default: 0.5
    medium: f32,   // default: 0.3
    slow: f32,     // default: 0.2
}
```

### Domain Events

```rust
/// Published when a pattern is promoted from fast to medium tier.
struct PatternPromoted {
    pattern_id: PatternId,
    from_tier: TemporalTier,
    to_tier: TemporalTier,
    persistence_duration: Duration,
    activation_count: u32,
}

/// Published when data is evicted from a tier.
struct TierEvicted {
    tier: TemporalTier,
    evicted_count: u64,
    reason: EvictionReason,
}

/// Published when long-term drift is detected.
struct DriftDetected {
    drift_magnitude: f32,       // optimal transport distance
    affected_channels: Vec<ChannelId>,
    drift_direction: Vec<f32>,  // manifold direction of drift
}
```

---

## Domain Services

### TemporalRouter

**Backed by:** `ruvector-router-core` v2.0.4, `ruvector-tiny-dancer-core` v2.0.4

Classifies each embedding into a temporal tier using a FastGRNN model. The router examines:
- **Inter-spike interval:** Short intervals (< 10 ms) indicate fast-tier spike bursts.
- **Session position:** Embeddings showing session-level trends are medium-tier.
- **Drift signature:** Embeddings that differ from the slow-tier baseline indicate drift.

**Performance contract:** < 10 microseconds per classification.

### BurstPatternDetector

Monitors the fast tier ring buffer for patterns that persist beyond the fast timescale (> 500 ms of repeated activation). These patterns are candidates for promotion to the medium tier.

### SonaConsolidator

**Backed by:** `ruvector-sona` v0.1.6

Handles medium-to-slow tier consolidation. Patterns that are stable across multiple sessions are compressed into drift vectors with EWC++ regularization to preserve existing slow-tier knowledge.

---

## Promotion & Eviction Rules

### Promotion

| From | To | Trigger | Data Transform |
|------|----|---------|----------------|
| Fast | Medium | Pattern persists > 500 ms with > 10 activations | Raw embeddings -> 8-bit quantized session tensor |
| Medium | Slow | Pattern stable across 3+ sessions | Session tensor -> 4-bit quantized drift vector via EWC++ consolidation |

### Eviction

| Tier | Trigger | Action |
|------|---------|--------|
| Fast | Ring buffer at capacity | Overwrite oldest entries (FIFO) |
| Medium | Session count exceeds `max_sessions` | Compress oldest sessions to 4-bit; merge into slow tier if stable |
| Slow | Memory pressure | Requantize from 4-bit to 2-bit; never fully evict |

---

## Invariants

1. **Exactly one tier.** Every embedding is classified into exactly one temporal tier. The router is deterministic for a given embedding + timing.
2. **Monotonic compression.** Data can only move from higher precision to lower precision (f32 -> 8-bit -> 4-bit -> 2-bit), never the reverse.
3. **Slow tier persistence.** Slow-tier data is never fully evicted; it may be requantized but the drift vector persists for the lifetime of the user profile.
4. **Budget compliance.** Total memory across all three tiers never exceeds the configured budget (derived from the 64 MB device constraint minus other subsystem allocations).

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| TemporalLearner | `synapse_graph::temporal` | (application layer) |
| TemporalRouter | `synapse_graph::temporal::router` | `ruvector-router-core` v2.0.4, `ruvector-tiny-dancer-core` v2.0.4 |
| TemporalRouter (WASM) | `synapse_graph::temporal::router_wasm` | `ruvector-tiny-dancer-wasm` v2.0.4 |
| FastTier | `synapse_graph::temporal::fast` | `ruvector-nervous-system` v2.0.4 |
| MediumTier | `synapse_graph::temporal::medium` | `ruvector-temporal-tensor` v2.0.4 |
| SlowTier | `synapse_graph::temporal::slow` | `ruvector-sona` v0.1.6 |
| DriftMeasurement | `synapse_graph::temporal::drift` | `ruvector-math` v2.0.4 |
