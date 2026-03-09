# DDD-005: Intent Prediction Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R6 | ADR-003, ADR-004
**Implementing Phase:** Phase 4

---

## Domain Overview

The Intent Prediction domain is the primary output-facing domain of SynapseGraph. It takes a partial neural pattern (an embedding from an incomplete thought), matches it against known cognitive shortcuts, applies temporal context weighting, and produces a ranked list of predicted intents with confidence scores.

This domain operates on the critical path with the strictest latency requirement: < 1 ms end-to-end from partial pattern to ranked prediction.

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Partial Pattern** | An embedding derived from an incomplete thought or neural activation. The system predicts intent from this fragment without waiting for the full pattern to complete. |
| **Intent** | A discrete user intention (e.g., "move left hand," "select item," "type letter A"). The vocabulary of intents is user-specific and grows over time. |
| **Intent Prediction** | A ranked list of (intent, confidence) pairs produced from a partial pattern. |
| **Shortcut Matching** | The process of comparing a partial pattern against registered cognitive shortcuts to find the most similar. |
| **Min-Cut Gated Attention** | An attention mechanism that uses dynamic graph min-cut as the gating function instead of softmax. Provides lower latency and better calibrated confidence scores for graph-structured data. |
| **Temporal Reweighting** | Adjusting prediction scores based on temporal context -- boosting predictions consistent with the user's recent (fast), session (medium), and long-term (slow) neural patterns. |
| **Prediction Feedback** | The signal received when a prediction is confirmed correct, corrected, or rejected. Drives the adaptation engine. |
| **Confidence Score** | A calibrated probability (0.0-1.0) that the predicted intent is correct. Derived from attention weights and temporal consistency. |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 INTENT PREDICTION CONTEXT                         |
|                                                                   |
|  Inbound:                                                         |
|    Partial pattern embedding (from Spike Ingestion, real-time)    |
|    CognitiveShortcut registry (from Cognitive Graph)              |
|    TemporalContext (from Temporal Learning)                       |
|    KNN results (from Vector Memory)                               |
|                                                                   |
|  +---------------------+                                         |
|  | IntentPredictor     |  Aggregate Root                         |
|  | (owns attention +   |  Orchestrates matching, attention,      |
|  |  scoring pipeline)  |  and temporal reweighting               |
|  +----------+----------+                                         |
|             |                                                     |
|    +--------+--------+---------+                                  |
|    |                 |         |                                   |
|    v                 v         v                                   |
|  +----------+  +----------+ +----------+                          |
|  | Shortcut |  | Attention| | Temporal |                          |
|  | Matcher  |  | Scorer   | | Reweight |                          |
|  +----------+  +----------+ +----------+                          |
|                                                                   |
|  Outbound:                                                        |
|    IntentPrediction --> [BCI output / user interface]              |
|    PredictionMade event --> [Adaptation Context]                   |
|    PredictionMade event --> [Provenance Context]                   |
|                                                                   |
+------------------------------------------------------------------+
```

### Context Map

| Relationship | Upstream | Downstream | Type |
|-------------|----------|------------|------|
| Cognitive Graph -> Intent Prediction | Cognitive Graph (DDD-004) | Intent Prediction | Published Language (CognitiveShortcut registry) |
| Temporal Learning -> Intent Prediction | Temporal Learning (DDD-003) | Intent Prediction | Open Host Service (TemporalContext API) |
| Vector Memory -> Intent Prediction | Vector Memory (DDD-002) | Intent Prediction | Open Host Service (KNN query API) |
| Intent Prediction -> Adaptation | Intent Prediction | Adaptation (DDD-005a) | Published Language (PredictionMade event) |
| Intent Prediction -> Provenance | Intent Prediction | Provenance (DDD-006) | Published Language (PredictionMade event) |

---

## Domain Model

### Aggregates

#### IntentPredictor (Aggregate Root)

```rust
/// Aggregate root for the intent prediction pipeline.
/// Invariant: predictions are always temporally contextualized.
/// Invariant: prediction latency < 1 ms end-to-end.
struct IntentPredictor {
    shortcut_matcher: ShortcutMatcher,
    attention_scorer: AttentionScorer,      // ruvector-attention + ruvector-attn-mincut
    temporal_reweighter: TemporalReweighter,
    intent_vocabulary: IntentVocabulary,
    config: PredictionConfig,
}

impl IntentPredictor {
    /// Predict intent from a partial neural pattern.
    /// Returns ranked predictions with confidence scores.
    fn predict(&self, partial_pattern: &GatedEmbedding,
               temporal_ctx: &TemporalContext)
        -> IntentPrediction;

    /// Record feedback on a prediction (correct, incorrect, corrected).
    fn record_feedback(&mut self, prediction_id: PredictionId,
                       feedback: PredictionFeedback)
        -> AdaptationSignal;
}
```

### Entities

#### IntentEntry

```rust
/// A known intent in the user's vocabulary.
/// Identity: intent_id.
struct IntentEntry {
    id: IntentId,
    label: String,                    // human-readable label
    shortcut_ids: Vec<ShortcutId>,    // associated cognitive shortcuts
    activation_count: u64,
    accuracy_rate: f32,               // historical prediction accuracy for this intent
    first_seen: Timestamp,
    last_predicted: Timestamp,
}
```

### Value Objects

#### IntentPrediction

```rust
/// The output of the prediction pipeline.
struct IntentPrediction {
    id: PredictionId,
    timestamp: Timestamp,
    partial_pattern_id: EmbeddingId,
    ranked_intents: Vec<ScoredIntent>,
    attention_weights: AttentionWeights,   // for interpretability
    temporal_influence: TierWeights,       // how much each tier contributed
    latency_us: u32,                      // actual prediction latency
}

struct ScoredIntent {
    intent_id: IntentId,
    confidence: f32,        // calibrated 0.0-1.0
    shortcut_id: ShortcutId,
    match_distance: f32,    // distance from partial pattern to shortcut
}
```

#### PredictionFeedback

```rust
/// Feedback on a prediction.
enum PredictionFeedback {
    Confirmed,                            // prediction was correct
    Corrected { actual: IntentId },       // user intended something else
    Rejected,                             // prediction was unwanted
    Timeout,                              // no feedback received within window
}
```

#### AdaptationSignal

```rust
/// Signal sent to the Adaptation domain when feedback indicates error.
struct AdaptationSignal {
    prediction_id: PredictionId,
    signal_type: AdaptationType,
    urgency: Urgency,
    partial_pattern: Vec<f32>,
    predicted_intent: IntentId,
    actual_intent: Option<IntentId>,
}

enum AdaptationType {
    PredictionError,    // wrong intent predicted
    ExplicitLabel,      // user provided correct label
    DriftDetected,      // pattern drift detected by temporal learning
}

enum Urgency {
    High,    // triggers fast LoRA (< 100us)
    Low,     // batched for slow LoRA (< 10ms)
}
```

### Domain Events

```rust
/// Published for every prediction made.
struct PredictionMade {
    prediction: IntentPrediction,
    feedback_window_ms: u32,   // how long to wait for feedback
}

/// Published when a new intent is discovered (user produces a pattern
/// not matching any known shortcuts).
struct NewIntentDiscovered {
    intent_id: IntentId,
    initial_pattern: Vec<f32>,
    discovery_context: String,
}

/// Published when prediction accuracy drops below threshold.
struct AccuracyDegraded {
    current_accuracy: f32,
    threshold: f32,
    window_size: u32,   // number of recent predictions in the window
}
```

---

## Domain Services

### ShortcutMatcher

Finds the cognitive shortcuts most similar to a partial pattern:

1. Query Vector Memory for KNN neighbors of the partial pattern.
2. Map KNN results to their associated cognitive shortcuts.
3. Rank shortcuts by match quality (distance + activation frequency).

### AttentionScorer

**Backed by:** `ruvector-attention` v2.0.4, `ruvector-attn-mincut` v2.0.4, `ruvector-attention-wasm` v2.0.4

Applies min-cut gated attention over matched shortcuts:

```
query   = partial_pattern
keys    = shortcut centroids
values  = intent associations
gate    = dynamic min-cut (not softmax)
```

Min-cut gated attention provides:
- **Lower latency** than softmax attention (no exp() computation).
- **Better calibration** for graph-structured data (attention weights respect graph topology).
- **Sparsity** (many attention weights are exactly zero, reducing computation).

### TemporalReweighter

Adjusts attention scores using the temporal context from DDD-003:

```
final_score = attention_score * temporal_consistency(prediction, temporal_context)
```

A prediction that is consistent with the user's recent fast-tier patterns and their session-level medium-tier trends receives a boost. A prediction that contradicts the temporal context is penalized.

---

## Prediction Pipeline (Step-by-Step)

```
1. Receive partial_pattern (GatedEmbedding)
2. ShortcutMatcher.match(partial_pattern)
   --> query VectorMemory.search_knn(partial_pattern, k=config.top_k)
   --> map results to CognitiveShortcuts via ShortcutRegistry
   --> return matched_shortcuts: Vec<(CognitiveShortcut, distance)>
3. AttentionScorer.score(partial_pattern, matched_shortcuts)
   --> apply min-cut gated attention
   --> return scored_intents: Vec<ScoredIntent>
4. TemporalReweighter.reweight(scored_intents, temporal_context)
   --> adjust scores for temporal consistency
   --> return reweighted_intents: Vec<ScoredIntent>
5. Sort by confidence, take top-k
6. Publish PredictionMade event
7. Return IntentPrediction
```

Total latency budget: 200 us (KNN) + 300 us (attention) + 100 us (reweight) + 100 us (overhead) < 1 ms.

---

## Cold Start Behavior

When the system has no cognitive shortcuts (new user, empty graph):

1. The `IntentPredictor` operates in **observation mode**.
2. All predictions return an empty ranked list with confidence 0.0.
3. The `NewIntentDiscovered` event is published for each unique pattern cluster.
4. Once the Cognitive Graph discovers shortcuts (Phase 3 completion), predictions begin.
5. The transition from observation to prediction mode is logged to the DAG.

---

## Invariants

1. **Temporal context required.** Every prediction must include temporal reweighting. Raw attention scores are never exposed to consumers.
2. **Latency budget.** Predictions exceeding 1 ms trigger a `LatencyBudgetExceeded` warning event. The pipeline must degrade gracefully (reduce k, skip temporal reweighting) rather than block.
3. **Calibrated confidence.** Confidence scores must be calibrated probabilities. The system tracks calibration error and recalibrates during slow LoRA consolidation.
4. **Feedback window.** Each prediction has a configurable feedback window. Predictions without feedback within the window are treated as `Timeout` (neither confirmed nor denied).

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| IntentPredictor | `synapse_graph::prediction` | (application layer) |
| ShortcutMatcher | `synapse_graph::prediction::matcher` | `ruvector-core` v2.0.5 (KNN), application layer |
| AttentionScorer | `synapse_graph::prediction::attention` | `ruvector-attention` v2.0.4, `ruvector-attn-mincut` v2.0.4 |
| AttentionScorer (WASM) | `synapse_graph::prediction::attention_wasm` | `ruvector-attention-wasm` v2.0.4 |
| TemporalReweighter | `synapse_graph::prediction::temporal` | (application layer, consumes TemporalContext) |
| IntentVocabulary | `synapse_graph::prediction::vocabulary` | (application layer) |
