# SynapseGraph -- SPARC Specification

> A Personalized Brain-Computer Interface Memory Layer powered by ruvector as the persistent, adaptive memory substrate for direct neural interfaces.

---

## S -- Specification

### Problem Statement

Current brain-computer interfaces (BCIs) such as Neuralink produce rich spike-train embeddings but have **no self-learning memory layer**. Every session starts from scratch. There is no system that:

1. Persistently stores and indexes neural embeddings across sessions.
2. Discovers personal cognitive "shortcuts" -- the neural encoding patterns unique to an individual user.
3. Learns across multiple timescales (spike bursts, session patterns, long-term neural drift).
4. Predicts intent before a full thought completes.
5. Runs entirely on a wearable device with sub-100 microsecond adaptation latency.

### Core Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| R1 | **Spike-Train Ingestion** | Ingest BCI spike-train embeddings (high-dimensional float vectors) at neural recording rates (30 kHz per channel, 1024+ channels). |
| R2 | **Persistent Vector Memory** | Store embeddings in an HNSW-indexed vector database with session, user, and temporal metadata. Survive device restarts. |
| R3 | **Cognitive Graph Construction** | Build a hypergraph of neural pattern relationships where nodes are embedding clusters and edges represent co-activation, temporal sequence, and semantic similarity. |
| R4 | **GNN Pattern Discovery** | Run a Graph Neural Network over the cognitive graph to discover personal neural encoding shortcuts -- recurring activation motifs unique to the user. |
| R5 | **Multi-Timescale Temporal Learning** | Learn across three tiers: **fast** (spike bursts, <10 ms), **medium** (session patterns, minutes to hours), **slow** (long-term neural drift, days to months). |
| R6 | **Intent Prediction** | Predict user intent from partial neural patterns using the learned cognitive shortcuts, before a full thought completes. |
| R7 | **On-Device LoRA Adaptation** | Adapt the model in real-time on-device using two-tier LoRA with Elastic Weight Consolidation (EWC++) to prevent catastrophic forgetting. Adaptation latency < 100 microseconds. |
| R8 | **WASM Wearable Deployment** | The full inference and learning pipeline must compile to WASM and run on a wearable edge device (ARM Cortex-M / RISC-V class). |
| R9 | **Coherence Gating** | Microsecond permit/deny decisions on memory writes to prevent hallucinated or corrupted embeddings from entering the persistent store. |
| R10 | **Provenance & Verifiability** | Tamper-evident DAG logging of neural event chains for auditability and debugging. |

### Non-Functional Requirements

- **Latency**: End-to-end ingestion-to-prediction < 1 ms. LoRA adaptation < 100 microseconds.
- **Memory**: Runtime footprint < 64 MB RAM on target wearable.
- **Storage**: Efficient quantized storage; temporal tensor compression for multi-timescale data.
- **Power**: Designed for battery-constrained wearable operation.
- **Privacy**: All processing on-device. No cloud dependency for core functionality.
- **Reliability**: Self-healing graph topology via dynamic min-cut analysis.

### Input / Output Contract

**Inputs:**
- Raw spike-train embeddings: `Vec<f32>` of dimension `d` (typically 256--1024) arriving at up to 30 kHz per channel.
- Channel metadata: electrode ID, spatial position, impedance.
- Session context: timestamp, user state labels (optional).

**Outputs:**
- Intent prediction: ranked list of predicted intents with confidence scores.
- Cognitive shortcut map: serializable graph of discovered personal neural patterns.
- Adaptation deltas: LoRA weight updates applied in-place.
- Provenance log: append-only DAG of neural events.

---

## P -- Pseudocode

### System Initialization

```
FUNCTION initialize_synapse_graph(config):
    // Core memory substrate
    vector_store     <- RuvectorCore.init(config.hnsw_params)
    cognitive_graph  <- RuvectorGraph.init(config.graph_params)
    collections      <- RuvectorCollections.init(vector_store)

    // Neural processing layers
    gnn_engine       <- RuvectorGNN.init(cognitive_graph.topology)
    attention        <- RuvectorAttention.init(config.attention_heads)
    nervous_system   <- RuvectorNervousSystem.init(config.spiking_params)

    // Temporal learning tiers
    temporal_tensor  <- RuvectorTemporalTensor.init(
        fast_window   = 10ms,
        medium_window = 1hour,
        slow_window   = 30days,
        quantization  = config.quant_bits
    )

    // On-device adaptation
    sona_engine      <- RuvectorSona.init(
        lora_rank     = config.lora_rank,
        ewc_lambda    = config.ewc_lambda,
        tier          = TwoTier  // fast-adapt + slow-consolidate
    )

    // Coherence and routing
    coherence_gate   <- CognitumGateKernel.init(config.gate_threshold)
    router           <- RuvectorRouterCore.init(config.routing_model)
    dag_log          <- RuvectorDAG.init(config.dag_params)

    RETURN SynapseGraph {
        vector_store, cognitive_graph, collections,
        gnn_engine, attention, nervous_system,
        temporal_tensor, sona_engine,
        coherence_gate, router, dag_log
    }
```

### Spike-Train Ingestion Pipeline

```
FUNCTION ingest_spike_train(sg: SynapseGraph, spike_batch: SpikeBatch):
    // Step 1: Bio-inspired preprocessing via spiking neural network
    processed <- sg.nervous_system.process_spikes(spike_batch)
    //   applies BTSP (Behavioral Time-Scale Plasticity) learning
    //   outputs: filtered embeddings + spike timing metadata

    // Step 2: Coherence gate -- reject corrupted / noisy embeddings
    FOR EACH embedding IN processed.embeddings:
        verdict <- sg.coherence_gate.evaluate(embedding)
        IF verdict == DENY:
            sg.dag_log.append(DenyEvent { embedding.id, reason: verdict.reason })
            CONTINUE

        // Step 3: Route to appropriate temporal tier
        tier <- sg.router.classify_temporal_tier(embedding, spike_batch.timing)
        //   FAST:   spike burst (< 10ms pattern)
        //   MEDIUM: session-level pattern
        //   SLOW:   long-term drift signature

        // Step 4: Store in vector memory with temporal metadata
        meta <- TemporalMeta {
            tier,
            timestamp:  spike_batch.timestamp,
            channel:    embedding.channel_id,
            session:    spike_batch.session_id
        }
        sg.vector_store.insert(embedding.vector, meta)

        // Step 5: Update temporal tensor at the appropriate timescale
        sg.temporal_tensor.accumulate(tier, embedding)

        // Step 6: Log provenance
        sg.dag_log.append(IngestEvent { embedding.id, tier, meta })
```

### Cognitive Graph Update

```
FUNCTION update_cognitive_graph(sg: SynapseGraph):
    // Runs periodically or after N ingestions

    // Step 1: Cluster recent embeddings
    clusters <- sg.vector_store.cluster_recent(
        method    = HNSW_NEIGHBORHOOD,
        threshold = config.cluster_threshold
    )

    // Step 2: Update hypergraph nodes and edges
    FOR EACH cluster IN clusters:
        node <- sg.cognitive_graph.upsert_node(cluster.centroid, cluster.metadata)

        // Co-activation edges
        co_active <- sg.vector_store.find_temporally_adjacent(cluster, window=10ms)
        FOR EACH neighbor IN co_active:
            sg.cognitive_graph.upsert_edge(
                node, neighbor.node,
                edge_type = CO_ACTIVATION,
                weight    = neighbor.correlation
            )

        // Temporal sequence edges
        sequences <- sg.temporal_tensor.extract_sequences(cluster, tier=MEDIUM)
        FOR EACH seq IN sequences:
            sg.cognitive_graph.upsert_edge(
                seq.from_node, seq.to_node,
                edge_type = TEMPORAL_SEQUENCE,
                weight    = seq.confidence
            )

    // Step 3: Self-healing via min-cut analysis
    weak_points <- sg.mincut.analyze(sg.cognitive_graph)
    FOR EACH weak_point IN weak_points:
        sg.cognitive_graph.reinforce(weak_point)
```

### GNN Cognitive Shortcut Discovery

```
FUNCTION discover_shortcuts(sg: SynapseGraph):
    // Step 1: Prepare graph for GNN
    adjacency  <- sg.cognitive_graph.to_adjacency()
    features   <- sg.cognitive_graph.node_features()

    // Step 2: Run graph transformer with proof-gated attention
    gnn_output <- sg.gnn_engine.forward(
        adjacency,
        features,
        attention_type = PROOF_GATED,     // ruvector-graph-transformer
        modules        = [TEMPORAL, MANIFOLD, BIOLOGICAL]
    )

    // Step 3: Extract cognitive shortcuts
    //   Shortcuts are subgraph motifs with high activation frequency
    //   and strong predictive power for subsequent activations
    shortcuts <- gnn_output.extract_motifs(
        min_frequency  = config.shortcut_min_freq,
        min_predictive = config.shortcut_min_pred,
        max_nodes      = config.shortcut_max_size
    )

    // Step 4: Register shortcuts for intent prediction
    FOR EACH shortcut IN shortcuts:
        sg.router.register_shortcut(shortcut)

    RETURN shortcuts
```

### Intent Prediction

```
FUNCTION predict_intent(sg: SynapseGraph, partial_pattern: Embedding):
    // Step 1: Find nearest cognitive shortcuts
    candidates <- sg.vector_store.search_knn(partial_pattern, k=config.top_k)

    // Step 2: Match against known shortcuts
    matched_shortcuts <- sg.router.match_shortcuts(candidates)

    // Step 3: Apply attention over matched shortcuts
    //   Uses min-cut gated attention for low-latency inference
    scored <- sg.attention.forward(
        query   = partial_pattern,
        keys    = matched_shortcuts.embeddings,
        values  = matched_shortcuts.intent_labels,
        gate    = MINCUT   // ruvector-attn-mincut
    )

    // Step 4: Temporal context weighting
    //   Boost predictions consistent with recent temporal context
    temporal_ctx <- sg.temporal_tensor.get_context(
        tiers = [FAST, MEDIUM, SLOW],
        weights = [0.5, 0.3, 0.2]   // fast tier dominates for real-time
    )
    scored <- scored.reweight(temporal_ctx)

    // Step 5: Return ranked predictions
    RETURN scored.top_k(config.prediction_k)
```

### On-Device LoRA Adaptation

```
FUNCTION adapt_online(sg: SynapseGraph, feedback: AdaptationSignal):
    // Runs when prediction accuracy drifts or explicit feedback arrives

    // Step 1: Compute adaptation target
    target <- MATCH feedback.type:
        PREDICTION_ERROR => compute_error_gradient(feedback)
        EXPLICIT_LABEL   => compute_supervised_gradient(feedback)
        DRIFT_DETECTED   => compute_drift_correction(feedback)

    // Step 2: Two-tier LoRA update via SONA
    sg.sona_engine.adapt(
        target,
        tier = IF feedback.urgency == HIGH THEN FAST_LORA
               ELSE SLOW_CONSOLIDATION,
        ewc_regularization = TRUE   // prevent catastrophic forgetting
    )
    //   FAST_LORA:           rank-4 update, < 100us
    //   SLOW_CONSOLIDATION:  rank-16 update, batched, with EWC++

    // Step 3: Propagate updates to GNN and attention layers
    sg.gnn_engine.apply_lora_delta(sg.sona_engine.last_delta())
    sg.attention.apply_lora_delta(sg.sona_engine.last_delta())

    // Step 4: Log adaptation event
    sg.dag_log.append(AdaptEvent {
        delta:    sg.sona_engine.last_delta().summary(),
        latency:  sg.sona_engine.last_latency(),
        tier:     feedback.type
    })
```

---

## A -- Architecture

### High-Level Architecture

```
+------------------------------------------------------------------+
|                    WEARABLE DEVICE (WASM Runtime)                 |
|                                                                   |
|  +------------------+    +-------------------+                    |
|  | BCI Hardware     |    | Coherence Gate    |                    |
|  | (Neuralink etc.) |--->| (cognitum-gate-   |                    |
|  | Spike Trains     |    |  kernel)          |                    |
|  +------------------+    +--------+----------+                    |
|                                   |                               |
|                          PERMIT   | DENY --> DAG Log              |
|                                   v                               |
|  +----------------------------------------------------------------+
|  |              INGESTION LAYER                                   |
|  |                                                                |
|  |  +---------------------+    +-------------------------+        |
|  |  | Nervous System      |    | Temporal Router         |        |
|  |  | (ruvector-nervous-  |    | (ruvector-router-core)  |        |
|  |  |  system-wasm)       |    |                         |        |
|  |  | - BTSP Learning     |--->| FAST  (< 10ms)          |        |
|  |  | - Spike Filtering   |    | MEDIUM (min-hrs)        |        |
|  |  +---------------------+    | SLOW  (days-months)     |        |
|  |                             +------------+------------+        |
|  +------------------------------|-----------|---------------------+
|                                 v           v                     |
|  +----------------------------------------------------------------+
|  |              MEMORY SUBSTRATE                                  |
|  |                                                                |
|  |  +---------------------+    +-------------------------+        |
|  |  | Vector Store        |    | Temporal Tensor         |        |
|  |  | (ruvector-core)     |    | (ruvector-temporal-     |        |
|  |  | - HNSW Index        |    |  tensor)                |        |
|  |  | - KNN Search        |    | - Tiered Quantization   |        |
|  |  | - Collections       |    | - 3-Scale Compression   |        |
|  |  +----------+----------+    +------------+------------+        |
|  |             |                            |                     |
|  +-------------|----------------------------|---------------------+
|                v                            v                     |
|  +----------------------------------------------------------------+
|  |              COGNITIVE GRAPH                                   |
|  |                                                                |
|  |  +---------------------+    +-------------------------+        |
|  |  | Hypergraph DB       |    | Min-Cut Self-Healing    |        |
|  |  | (ruvector-graph)    |    | (ruvector-mincut)       |        |
|  |  | - Co-activation     |    | - Topology Repair       |        |
|  |  | - Temporal Sequence  |    | - Weak Point Detection |        |
|  |  | - Semantic Similarity|    |                         |        |
|  |  +----------+----------+    +-------------------------+        |
|  +-------------|----------------------------------------------+   |
|                v                                                  |
|  +----------------------------------------------------------------+
|  |              INTELLIGENCE LAYER                                |
|  |                                                                |
|  |  +---------------------+    +-------------------------+        |
|  |  | GNN Engine          |    | Attention               |        |
|  |  | (ruvector-gnn-wasm) |    | (ruvector-attention-    |        |
|  |  | - Shortcut Discovery|    |  wasm)                  |        |
|  |  | - Graph Transformer |    | - Min-Cut Gated         |        |
|  |  | - Proof-Gated Attn  |    | - Geometric / Sparse    |        |
|  |  +----------+----------+    +------------+------------+        |
|  |             |                            |                     |
|  +-------------|----------------------------|---------------------+
|                v                            v                     |
|  +----------------------------------------------------------------+
|  |              PREDICTION & ADAPTATION                           |
|  |                                                                |
|  |  +---------------------+    +-------------------------+        |
|  |  | Intent Predictor    |    | SONA LoRA Engine        |        |
|  |  | - Shortcut Matching |    | (ruvector-sona)         |        |
|  |  | - Temporal Context  |    | - Two-Tier LoRA         |        |
|  |  | - Ranked Output     |    | - EWC++ Regularization  |        |
|  |  +---------------------+    | - < 100us Adaptation    |        |
|  |                             +-------------------------+        |
|  +----------------------------------------------------------------+
|                                                                   |
|  +---------------------------+                                    |
|  | DAG Provenance Log        |                                    |
|  | (ruvector-dag)            |                                    |
|  | - Tamper-Evident Chain    |                                    |
|  | - Neural Event History    |                                    |
|  +---------------------------+                                    |
+------------------------------------------------------------------+
```

### Dependency Graph

```
Layer 0 (Foundation):
    ruvector-core, ruvector-math, ruvector-collections

Layer 1 (Graph & Temporal):
    ruvector-graph        --> ruvector-core
    ruvector-temporal-tensor --> ruvector-core, ruvector-math
    ruvector-dag          --> ruvector-core

Layer 2 (Neural Processing):
    ruvector-nervous-system --> ruvector-core, ruvector-math
    ruvector-gnn            --> ruvector-graph, ruvector-core
    ruvector-graph-transformer --> ruvector-gnn, ruvector-attention
    ruvector-attention      --> ruvector-math
    ruvector-attn-mincut    --> ruvector-attention, ruvector-mincut
    ruvector-mincut         --> ruvector-graph

Layer 3 (Routing & Adaptation):
    ruvector-router-core    --> ruvector-core
    ruvector-sona           --> ruvector-core, ruvector-math
    cognitum-gate-kernel    --> (no_std, standalone)
    ruv-fann                --> (standalone neural engine)

Layer 4 (WASM Bindings):
    ruvector-wasm                --> ruvector-core
    ruvector-gnn-wasm            --> ruvector-gnn
    ruvector-attention-wasm      --> ruvector-attention
    ruvector-nervous-system-wasm --> ruvector-nervous-system
    ruvector-learning-wasm       --> ruvector-sona
    ruvector-tiny-dancer-core    --> ruvector-router-core
    ruvector-tiny-dancer-wasm    --> ruvector-tiny-dancer-core
```

### Module Responsibilities

| Module | Crate(s) | Responsibility |
|--------|----------|----------------|
| **SpikeIngester** | `ruvector-nervous-system-wasm` | Receives raw spike trains, applies BTSP learning, filters noise. |
| **CoherenceGate** | `cognitum-gate-kernel` | No-std microsecond permit/deny on each embedding before storage. |
| **TemporalRouter** | `ruvector-router-core`, `ruvector-tiny-dancer-core` | Classifies each embedding into fast/medium/slow temporal tier. |
| **VectorMemory** | `ruvector-core`, `ruvector-collections` | HNSW-indexed persistent vector store. |
| **TemporalTensor** | `ruvector-temporal-tensor` | Tiered quantized storage with temporal compression across 3 timescales. |
| **CognitiveGraph** | `ruvector-graph`, `ruvector-mincut` | Hypergraph of neural pattern relationships with self-healing topology. |
| **ShortcutDiscovery** | `ruvector-gnn`, `ruvector-graph-transformer` | GNN-based discovery of personal cognitive encoding motifs. |
| **IntentPredictor** | `ruvector-attention`, `ruvector-attn-mincut` | Low-latency intent prediction from partial neural patterns. |
| **AdaptationEngine** | `ruvector-sona`, `ruvector-learning-wasm` | Two-tier LoRA with EWC++ for on-device model adaptation. |
| **ProvenanceLog** | `ruvector-dag` | Tamper-evident DAG of all neural events and adaptation decisions. |
| **MathKernel** | `ruvector-math` | Optimal transport, information geometry for drift measurement. |
| **SparseInference** | `ruvector-sparse-inference`, `ruv-fann` | PowerInfer-style sparse inference and CPU-native neural engine. |

### Data Flow

```
Spike Train --> SpikeIngester --> CoherenceGate --PERMIT--> TemporalRouter
                                      |                        |
                                    DENY                  +---------+
                                      |                   |  FAST   |
                                      v                   |  MEDIUM |
                                 ProvenanceLog            |  SLOW   |
                                      ^                   +---------+
                                      |                     |     |
                              (all events)          +-------+     |
                                      |             v             v
                                      +--- VectorMemory   TemporalTensor
                                      |         |               |
                                      |         v               v
                                      +--- CognitiveGraph <----+
                                      |         |
                                      |         v
                                      +--- ShortcutDiscovery (GNN)
                                      |         |
                                      |         v
                                      +--- IntentPredictor
                                      |         |
                                      |    prediction + confidence
                                      |         |
                                      |         v
                                      +--- AdaptationEngine (LoRA)
                                                |
                                           weight deltas propagated
                                           back to GNN + Attention
```

### External Interfaces

**JavaScript / TypeScript Layer (optional):**
- `ruvector` (npm) -- Node.js entry point to memory layer
- `@ruvector/gnn` -- GNN bindings for JS orchestration
- `@ruvector/rvf-wasm` -- Browser/edge WASM vector operations
- `@ruvector/rvf` -- Unified TypeScript SDK
- `ruvector-extensions` -- Temporal tracking additions
- `midstreamer` -- WASM temporal analysis (DTW, LCS)

**Agent Orchestration Layer (optional):**
- `agentic-flow` -- Multi-agent coordination with ReasoningBank
- `@claude-flow/plugin-neural-coordination` -- SONA + GNN multi-agent plugin
- `@claude-flow/plugin-cognitive-kernel` -- Working memory and meta-cognition

**Distributed Layer (optional):**
- `ruvnet/Synaptic-Mesh` -- P2P DAG substrate for multi-device coordination
- `ruvnet/QuDAG` -- Tamper-evident verifiable logging via DAG consensus
- `ruvnet/dspy.ts` -- Declarative DSL for cognitive pattern configuration

---

## R -- Refinement

### Critical Design Decisions

#### 1. Three-Tier Temporal Architecture

The temporal learning system is the most architecturally significant decision. Rather than a single model that tries to learn at all timescales, we decompose into three explicit tiers:

| Tier | Timescale | Data Structure | Update Strategy | Crate |
|------|-----------|----------------|-----------------|-------|
| **Fast** | < 10 ms | Ring buffer of raw spike embeddings | BTSP (Behavioral Time-Scale Plasticity) | `ruvector-nervous-system` |
| **Medium** | Minutes to hours | Session-level compressed tensors | Batched GNN re-training on graph deltas | `ruvector-temporal-tensor` |
| **Slow** | Days to months | Quantized long-term drift vectors | EWC++ regularized consolidation | `ruvector-sona` |

**Rationale:** Neural signals have fundamentally different information content at each timescale. A spike burst carries motor intent; a session pattern reveals learning progress; long-term drift tracks neuroplasticity. Separate tiers prevent fast-timescale noise from corrupting slow-timescale models.

**Interaction:** The `TemporalRouter` (backed by `ruvector-tiny-dancer-core` FastGRNN) classifies each embedding into a tier. The `TemporalTensor` handles cross-tier aggregation: fast-tier patterns that persist get promoted to medium; medium patterns that are stable across sessions get consolidated into slow.

#### 2. Coherence Gating as Hard Boundary

The `cognitum-gate-kernel` runs as a no-std WASM module that evaluates every embedding before it enters the memory substrate. This is a hard architectural boundary, not a soft filter.

**Why:** BCI signals are inherently noisy. Electrode impedance changes, movement artifacts, and neural recording dropouts can produce embeddings that look valid but carry no cognitive content. Admitting these into the HNSW index degrades search quality and pollutes the cognitive graph. The coherence gate uses a fast statistical test (manifold consistency check via `ruvector-math`) to reject outliers in < 1 microsecond.

#### 3. Proof-Gated Graph Attention

The `ruvector-graph-transformer` provides 8 verified attention modules. For SynapseGraph we use three:

- **Temporal:** Attention weighted by recency and temporal proximity of neural events.
- **Manifold:** Attention weighted by position on the neural embedding manifold (captures topological structure).
- **Biological:** Attention weighted by known neuroanatomical connectivity priors (e.g., cortical column adjacency).

**Why proof-gated:** Each attention computation produces a verifiable proof that the output is consistent with the input graph structure. This prevents adversarial or corrupted graph states from producing confident but wrong predictions.

#### 4. Two-Tier LoRA for On-Device Adaptation

SONA provides a two-tier LoRA design:

- **Fast LoRA (rank-4):** Applied immediately on prediction error. Adapts attention heads and the final classification layer. Target: < 100 microseconds.
- **Slow LoRA (rank-16):** Applied in batches during idle periods. Adapts GNN message-passing weights with EWC++ regularization. Target: < 10 milliseconds.

**Why two tiers:** A single LoRA rank is a poor fit. Low rank adapts quickly but has limited expressiveness; high rank is expressive but too slow for real-time and risks catastrophic forgetting. The two-tier design matches the temporal hierarchy: fast LoRA handles real-time intent correction, slow LoRA handles long-term model evolution.

#### 5. Self-Healing Graph via Dynamic Min-Cut

The `ruvector-mincut` crate provides subpolynomial dynamic min-cut analysis. We run this periodically on the cognitive graph to:

- Detect weak points (bottleneck edges) that could fragment the graph.
- Reinforce weak points by adding redundant edges from the temporal tensor history.
- Prune dead regions (clusters with no recent activations beyond slow-tier threshold).

**Why:** The cognitive graph grows continuously. Without maintenance, it develops bottlenecks (single edges connecting large subgraphs) that make shortcut discovery brittle. Min-cut analysis keeps the graph well-connected and resilient.

### Edge Cases and Failure Modes

| Scenario | Mitigation |
|----------|------------|
| Electrode failure (channel dropout) | Coherence gate rejects; router redistributes load to surviving channels. Slow-tier model retains learned patterns from the lost channel. |
| Sudden neural pattern shift (e.g., seizure, medication change) | Drift detection triggers SONA fast-LoRA reset for affected subgraph region. EWC++ preserves unaffected regions. Provenance DAG logs the event for post-hoc analysis. |
| Memory pressure on wearable | Temporal tensor applies aggressive quantization to slow-tier data. Oldest fast-tier data is evicted from ring buffer. HNSW index parameters are dynamically reduced. |
| Adversarial embedding injection | Coherence gate's manifold consistency check rejects out-of-distribution inputs. Proof-gated attention prevents graph corruption even if an adversarial embedding bypasses the gate. |
| Cold start (new user, empty graph) | System operates in "observation mode" for initial sessions: ingests and stores but does not predict. GNN runs shortcut discovery after minimum data threshold is reached. |

### Performance Budget

| Operation | Target Latency | Crate |
|-----------|---------------|-------|
| Spike ingestion + BTSP | < 100 us | `ruvector-nervous-system-wasm` |
| Coherence gate decision | < 1 us | `cognitum-gate-kernel` |
| Temporal routing | < 10 us | `ruvector-tiny-dancer-wasm` |
| HNSW insert | < 50 us | `ruvector-wasm` |
| HNSW KNN search (k=10) | < 200 us | `ruvector-wasm` |
| GNN shortcut discovery | < 500 ms (batched, async) | `ruvector-gnn-wasm` |
| Intent prediction (full pipeline) | < 1 ms | `ruvector-attention-wasm` |
| Fast LoRA adaptation | < 100 us | `ruvector-learning-wasm` |
| Slow LoRA consolidation | < 10 ms (batched, async) | `ruvector-learning-wasm` |
| Min-cut graph analysis | < 100 ms (periodic, async) | `ruvector-mincut` |
| DAG log append | < 5 us | `ruvector-dag` |

---

## C -- Completion

### Implementation Phases

#### Phase 1: Foundation (Vector Memory + Spike Ingestion)

**Goal:** Establish the core data pipeline from BCI spike trains to persistent vector storage.

**Tasks:**
1. Set up Rust workspace with `ruvector-core`, `ruvector-collections`, `ruvector-math` as foundation crates.
2. Implement `SpikeIngester` module wrapping `ruvector-nervous-system` with BTSP learning enabled.
3. Implement `CoherenceGate` module wrapping `cognitum-gate-kernel` with manifold consistency check.
4. Implement `VectorMemory` module wrapping `ruvector-core` HNSW with temporal metadata schema.
5. Build WASM bindings for the ingestion pipeline using `ruvector-wasm` and `ruvector-nervous-system-wasm`.
6. Write spike-train simulator for testing (synthetic spike data with known patterns).
7. Benchmark: ingestion throughput > 30k embeddings/sec, single insert < 50 us.

**Deliverable:** Spike trains flow from simulated BCI through coherence gate into persistent HNSW store.

#### Phase 2: Temporal Learning

**Goal:** Implement the three-tier temporal architecture.

**Tasks:**
1. Integrate `ruvector-temporal-tensor` with tiered quantization configured for fast/medium/slow windows.
2. Implement `TemporalRouter` using `ruvector-router-core` and `ruvector-tiny-dancer-core` FastGRNN.
3. Build tier promotion logic: fast -> medium (persistence detection), medium -> slow (cross-session stability).
4. Build tier eviction logic: fast ring buffer overflow, slow-tier quantization under memory pressure.
5. WASM bindings for temporal router via `ruvector-tiny-dancer-wasm`.
6. Test with synthetic spike trains containing known multi-timescale patterns.
7. Benchmark: routing decision < 10 us, temporal accumulation < 50 us.

**Deliverable:** Embeddings are classified into temporal tiers and stored with appropriate compression.

#### Phase 3: Cognitive Graph + GNN

**Goal:** Build the hypergraph of neural patterns and discover cognitive shortcuts.

**Tasks:**
1. Integrate `ruvector-graph` as the hypergraph substrate.
2. Implement cognitive graph construction: clustering, co-activation edges, temporal sequence edges.
3. Integrate `ruvector-mincut` for self-healing graph maintenance.
4. Integrate `ruvector-gnn` and `ruvector-graph-transformer` for shortcut discovery.
5. Configure proof-gated attention with temporal, manifold, and biological modules.
6. Implement motif extraction algorithm for identifying cognitive shortcuts.
7. WASM bindings via `ruvector-gnn-wasm`.
8. Test with synthetic data containing planted motifs; verify discovery accuracy > 95%.

**Deliverable:** System discovers and registers cognitive shortcuts from the graph of neural patterns.

#### Phase 4: Intent Prediction + LoRA Adaptation

**Goal:** Real-time intent prediction with on-device model adaptation.

**Tasks:**
1. Implement `IntentPredictor` using `ruvector-attention` with min-cut gated attention.
2. Implement temporal context weighting across all three tiers.
3. Integrate `ruvector-sona` for two-tier LoRA adaptation.
4. Implement fast-LoRA path (rank-4, < 100 us) for real-time correction.
5. Implement slow-LoRA path (rank-16, < 10 ms, batched) with EWC++ regularization.
6. Wire adaptation feedback loop: prediction error -> LoRA update -> model refresh.
7. WASM bindings via `ruvector-attention-wasm` and `ruvector-learning-wasm`.
8. Benchmark: end-to-end prediction < 1 ms, fast-LoRA < 100 us.

**Deliverable:** System predicts intent from partial patterns and adapts in real-time.

#### Phase 5: Provenance, Integration & Hardening

**Goal:** Production-ready system with full provenance, DAG logging, and integration testing.

**Tasks:**
1. Integrate `ruvector-dag` for tamper-evident neural event logging.
2. Implement provenance queries: "why was this intent predicted?", "when did this shortcut emerge?"
3. Build JavaScript orchestration layer using `ruvector` (npm), `@ruvector/gnn`, `@ruvector/rvf-wasm`.
4. Integrate `ruv-fann` as the CPU-native neural engine fallback for non-WASM targets.
5. Integrate `ruvector-sparse-inference` for PowerInfer-style sparse execution on resource-constrained devices.
6. End-to-end integration tests with realistic synthetic BCI data.
7. Memory profiling: verify < 64 MB runtime footprint.
8. Power profiling on target wearable hardware (or emulation).
9. Optional: integrate `ruvnet/Synaptic-Mesh` + `ruvnet/QuDAG` for multi-device coordination.
10. Optional: integrate `agentic-flow` + `@claude-flow/plugin-neural-coordination` for agent orchestration.

**Deliverable:** Production-ready SynapseGraph running on wearable WASM runtime with full provenance.

### Verification & Testing Strategy

| Level | Scope | Method |
|-------|-------|--------|
| **Unit** | Individual crate functions | Rust `#[test]` with property-based testing (`proptest`) for vector operations. |
| **Integration** | Cross-module data flow | Spike simulator -> full pipeline -> verify stored embeddings and graph structure. |
| **Temporal** | Multi-timescale learning | Synthetic spike trains with planted patterns at known timescales; verify tier classification and promotion accuracy. |
| **GNN** | Shortcut discovery | Synthetic graphs with planted motifs; verify precision/recall > 95%. |
| **Adaptation** | LoRA convergence | Introduce controlled distribution shift; verify fast-LoRA corrects within 10 iterations, slow-LoRA stabilizes within 100. |
| **Latency** | Performance budgets | Criterion.rs benchmarks for every critical-path operation. CI gates on regression > 10%. |
| **Memory** | Footprint constraints | WASM build size < 4 MB. Runtime RSS < 64 MB under sustained 30 kHz ingestion. |
| **Adversarial** | Robustness | Fuzz coherence gate with random / adversarial embeddings; verify 0% leakage of out-of-distribution inputs. |
| **Provenance** | DAG integrity | Verify DAG hash chain after 1M events; no unlinked or corrupted nodes. |

### Minimal Viable Configuration

For initial development and testing, the following subset is sufficient:

**Hard Dependencies (Rust):**
```toml
[dependencies]
ruvector-core = "2.0.5"
ruvector-gnn = "2.0.5"
ruvector-nervous-system = "2.0.4"
ruvector-sona = "0.1.6"
ruvector-temporal-tensor = "2.0.4"
ruvector-attention = "2.0.4"
ruvector-graph = "2.0.4"
ruvector-wasm = "0.1.29"
ruvector-nervous-system-wasm = "2.0.4"
ruvector-learning-wasm = "2.0.4"
ruvector-gnn-wasm = "2.0.4"
cognitum-gate-kernel = "0.1.1"
ruv-fann = "0.2.0"
```

**Hard Dependencies (npm):**
```json
{
  "dependencies": {
    "ruvector": "^0.2.11",
    "@ruvector/gnn": "^0.1.25",
    "@ruvector/rvf-wasm": "^0.1.6"
  }
}
```

**Required Repos:**
- `ruvnet/ruvector` -- core workspace (mandatory)
- `ruvnet/ruv-FANN` -- WASM neural runtime (mandatory)

### Success Criteria

1. **Functional:** System ingests synthetic spike trains, builds cognitive graph, discovers planted shortcuts, and predicts intent with > 80% accuracy on test data.
2. **Latency:** End-to-end ingestion-to-prediction < 1 ms. LoRA adaptation < 100 us. Coherence gate < 1 us.
3. **Memory:** Runtime footprint < 64 MB on WASM target.
4. **Adaptation:** System improves prediction accuracy by > 20% over 100 sessions of synthetic use without catastrophic forgetting of earlier patterns.
5. **Provenance:** Every neural event and adaptation decision is traceable through the DAG log.
6. **Novelty:** No existing BCI system provides a self-learning memory layer. SynapseGraph is the first.
