# DDD-006: Adaptation Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R7 | ADR-004
**Implementing Phase:** Phase 4

---

## Domain Overview

The Adaptation domain owns the on-device model adaptation lifecycle. When predictions are wrong, when neural patterns drift, or when explicit feedback arrives, this domain computes and applies LoRA weight updates to the GNN and attention layers. It implements the two-tier LoRA design from ADR-004 with EWC++ regularization to prevent catastrophic forgetting.

This domain has the tightest latency constraint in the system: fast LoRA must complete in < 100 microseconds.

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **LoRA** | Low-Rank Adaptation. A parameter-efficient fine-tuning method that decomposes weight updates into low-rank matrices (A * B), dramatically reducing the number of trainable parameters. |
| **Fast LoRA** | Rank-4 LoRA applied immediately on prediction error. Modifies attention heads and classification layer only. Target: < 100 us. Ephemeral (not persisted). |
| **Slow LoRA** | Rank-16 LoRA applied in batches during idle periods. Modifies all layers including GNN message-passing. Target: < 10 ms. Persistent (survives restarts). |
| **EWC++** | Elastic Weight Consolidation (enhanced). Regularization that penalizes changes to weights important for previously learned tasks. Prevents catastrophic forgetting. |
| **Fisher Information Matrix** | A matrix estimating the importance of each weight for current task performance. Used by EWC++ to protect important weights. |
| **LoRA Delta** | The weight update computed by a LoRA adaptation step. Represented as a pair of low-rank matrices (A, B) where the full update is A * B. |
| **Adaptation Signal** | The trigger for adaptation, produced by Intent Prediction or Temporal Learning domains when errors or drift are detected. |
| **Consolidation** | The process of absorbing accumulated fast LoRA deltas into the slow LoRA, running EWC++ to protect important weights. |
| **Catastrophic Forgetting** | The phenomenon where learning new patterns destroys previously learned patterns. The primary risk this domain mitigates. |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 ADAPTATION CONTEXT                                 |
|                                                                   |
|  Inbound:                                                         |
|    AdaptationSignal (from Intent Prediction)                      |
|    DriftDetected event (from Temporal Learning)                   |
|    Graph topology (from Cognitive Graph)                          |
|                                                                   |
|  +---------------------+                                         |
|  | AdaptationEngine    |  Aggregate Root                         |
|  | (owns SONA +        |  Orchestrates two-tier LoRA,            |
|  |  fast/slow paths)   |  EWC++, and delta propagation           |
|  +----------+----------+                                         |
|             |                                                     |
|    +--------+--------+                                            |
|    |                 |                                            |
|    v                 v                                            |
|  +----------+  +------------------+                               |
|  | FastLoRA |  | SlowLoRA         |                               |
|  | Path     |  | Path + EWC++     |                               |
|  | (rank-4) |  | (rank-16)        |                               |
|  +----------+  +------------------+                               |
|                                                                   |
|  Outbound:                                                        |
|    LoRA deltas --> [GNN Engine, Attention Scorer]                  |
|    AdaptationApplied event --> [Provenance Context]                |
|    Consolidated event --> [Provenance Context]                     |
|                                                                   |
+------------------------------------------------------------------+
```

### Context Map

| Relationship | Upstream | Downstream | Type |
|-------------|----------|------------|------|
| Intent Prediction -> Adaptation | Intent Prediction (DDD-005) | Adaptation | Published Language (AdaptationSignal) |
| Temporal Learning -> Adaptation | Temporal Learning (DDD-003) | Adaptation | Published Language (DriftDetected event) |
| Adaptation -> Cognitive Graph | Adaptation | Cognitive Graph (DDD-004) | Shared Kernel (LoRA deltas applied to GNN weights) |
| Adaptation -> Intent Prediction | Adaptation | Intent Prediction (DDD-005) | Shared Kernel (LoRA deltas applied to attention weights) |
| Adaptation -> Provenance | Adaptation | Provenance (DDD-006a) | Published Language (adaptation events) |

---

## Domain Model

### Aggregates

#### AdaptationEngine (Aggregate Root)

```rust
/// Aggregate root for the on-device adaptation system.
/// Invariant: fast LoRA completes in < 100 microseconds.
/// Invariant: slow LoRA applies EWC++ regularization.
/// Invariant: all adaptations are logged to provenance.
struct AdaptationEngine {
    sona: SonaEngine,                    // ruvector-sona
    fast_path: FastLoraPath,
    slow_path: SlowLoraPath,
    fisher_matrix: FisherInformationMatrix,
    pending_signals: Vec<AdaptationSignal>,
    config: AdaptationConfig,
}

impl AdaptationEngine {
    /// Process an adaptation signal (immediate for High urgency, queued for Low).
    fn adapt(&mut self, signal: AdaptationSignal) -> AdaptationResult;

    /// Run slow LoRA consolidation (called during idle periods).
    fn consolidate(&mut self) -> ConsolidationResult;

    /// Get the most recent LoRA delta for propagation to downstream modules.
    fn last_delta(&self) -> &LoraDelta;

    /// Get the latency of the most recent adaptation.
    fn last_latency(&self) -> Duration;

    /// Recompute the Fisher Information Matrix from current model state.
    fn recompute_fisher(&mut self, graph: &CognitiveGraph);
}
```

### Entities

#### FastLoraPath

```rust
/// The fast LoRA adaptation path.
/// Identity: singleton within AdaptationEngine.
struct FastLoraPath {
    rank: usize,                   // 4
    target_layers: Vec<LayerId>,   // attention heads + classification
    accumulated_deltas: Vec<LoraDelta>,
    damping_factor: f32,           // exponential moving average for oscillation prevention
    max_accumulated: usize,        // max deltas before forcing consolidation
}
```

#### SlowLoraPath

```rust
/// The slow LoRA adaptation path with EWC++ regularization.
/// Identity: singleton within AdaptationEngine.
struct SlowLoraPath {
    rank: usize,                   // 16
    target_layers: Vec<LayerId>,   // all layers (GNN + attention + classification)
    ewc_lambda: f32,               // regularization strength
    persisted_deltas: Vec<LoraDelta>,
    last_consolidation: Timestamp,
    consolidation_count: u64,
}
```

### Value Objects

#### LoraDelta

```rust
/// A low-rank weight update: delta_W = A * B^T
struct LoraDelta {
    layer_id: LayerId,
    matrix_a: Vec<f32>,   // shape: (d_out, rank)
    matrix_b: Vec<f32>,   // shape: (d_in, rank)
    rank: usize,
    computed_at: Timestamp,
    trigger: AdaptationType,
}

impl LoraDelta {
    /// Compute the full weight update (for small layers).
    fn to_full_matrix(&self) -> Vec<Vec<f32>>;

    /// Approximate memory size in bytes.
    fn memory_bytes(&self) -> usize;

    /// Summary for provenance logging.
    fn summary(&self) -> DeltaSummary;
}
```

#### FisherInformationMatrix

```rust
/// Diagonal approximation of the Fisher Information Matrix.
/// Tracks which weights are important for previously learned shortcuts.
struct FisherInformationMatrix {
    diagonal: Vec<f32>,   // one importance score per weight
    computed_at: Timestamp,
    sample_count: u64,    // number of samples used to estimate
}
```

#### AdaptationResult

```rust
/// Result of a single adaptation step.
struct AdaptationResult {
    delta: LoraDelta,
    latency: Duration,
    path: LoraPath,         // Fast or Slow
    ewc_penalty: f32,       // 0.0 for fast path; regularization cost for slow path
}

enum LoraPath { Fast, Slow }
```

#### ConsolidationResult

```rust
/// Result of slow LoRA consolidation.
struct ConsolidationResult {
    deltas_consolidated: u32,
    total_latency: Duration,
    ewc_penalty: f32,
    weights_protected: u32,    // weights where EWC++ prevented large changes
    fisher_recomputed: bool,
}
```

### Domain Events

```rust
/// Published after every adaptation (fast or slow).
struct AdaptationApplied {
    result: AdaptationResult,
    affected_layers: Vec<LayerId>,
    trigger: AdaptationType,
}

/// Published after slow LoRA consolidation.
struct Consolidated {
    result: ConsolidationResult,
    accumulated_fast_deltas: u32,  // how many fast deltas were absorbed
}

/// Published when fast LoRA oscillation is detected.
struct OscillationDetected {
    layer_id: LayerId,
    oscillation_magnitude: f32,
    damping_applied: f32,
}

/// Published when Fisher matrix is recomputed.
struct FisherRecomputed {
    sample_count: u64,
    top_protected_weights: Vec<(LayerId, usize, f32)>,  // most important weights
}
```

---

## Domain Services

### SonaEngine

**Backed by:** `ruvector-sona` v0.1.6, `ruvector-learning-wasm` v2.0.4

The core LoRA computation engine. Provides:
- Rank-4 and rank-16 LoRA forward/backward computation.
- EWC++ regularization with diagonal Fisher approximation.
- Two-tier scheduling: fast path (immediate) and slow path (batched).
- WASM-compiled for on-device execution.

### Delta Propagation

After a LoRA delta is computed, it must be propagated to the affected modules:

```rust
// Fast LoRA delta propagation
gnn_engine.apply_lora_delta(delta);       // if layer is a GNN layer
attention_scorer.apply_lora_delta(delta);   // if layer is an attention layer

// These are shared kernel interactions with the Cognitive Graph and
// Intent Prediction bounded contexts.
```

### Oscillation Damping

Fast LoRA can oscillate when conflicting signals arrive rapidly. The damping mechanism:

1. Compute exponential moving average of recent deltas for each layer.
2. If the delta sign flips more than N times in M predictions, apply damping.
3. Damping reduces the learning rate for the oscillating layer until stability returns.
4. Publish `OscillationDetected` event for monitoring.

---

## Adaptation Pipeline

### Fast Path (< 100 us)

```
1. Receive AdaptationSignal with urgency=High
2. Compute error gradient from signal
3. Compute rank-4 LoRA delta: A(4 x d_out) * B(4 x d_in)^T
4. Apply damping if oscillation detected
5. Apply delta to attention heads and classification layer
6. Append delta to accumulated_deltas
7. Publish AdaptationApplied event
8. If accumulated_deltas.len() > max_accumulated:
     trigger slow consolidation
```

### Slow Path (< 10 ms)

```
1. Collect all accumulated fast deltas + any queued Low-urgency signals
2. Compute Fisher Information Matrix (or reuse if recent)
3. For each target layer:
   a. Compute rank-16 LoRA delta
   b. Apply EWC++ penalty: loss += lambda * sum(F_i * (theta_i - theta_i*)^2)
   c. Clip delta if EWC penalty exceeds threshold
4. Apply deltas to all layers (GNN, attention, classification)
5. Persist deltas to device storage
6. Clear accumulated fast deltas
7. Publish Consolidated event
```

---

## Invariants

1. **Fast path latency.** Fast LoRA must complete in < 100 us. If a computation would exceed this, it is deferred to the slow path.
2. **EWC++ on slow path.** Every slow LoRA update applies EWC++ regularization. There is no unregularized slow path.
3. **Delta persistence.** Slow LoRA deltas are persisted before the `Consolidated` event is published. Device restart restores the last consolidated state.
4. **Provenance completeness.** Every adaptation (fast and slow) produces a provenance event with delta summary, latency, and trigger information.
5. **Oscillation safety.** If oscillation is detected, the fast path's learning rate is reduced. The system never amplifies oscillation.

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| AdaptationEngine | `synapse_graph::adaptation` | (application layer) |
| SonaEngine | `synapse_graph::adaptation::sona` | `ruvector-sona` v0.1.6 |
| SonaEngine (WASM) | `synapse_graph::adaptation::sona_wasm` | `ruvector-learning-wasm` v2.0.4 |
| FastLoraPath | `synapse_graph::adaptation::fast` | `ruvector-sona` v0.1.6 |
| SlowLoraPath | `synapse_graph::adaptation::slow` | `ruvector-sona` v0.1.6 |
| FisherInformationMatrix | `synapse_graph::adaptation::fisher` | `ruvector-math` v2.0.4 |
| Delta Propagation | `synapse_graph::adaptation::propagation` | (application layer) |
