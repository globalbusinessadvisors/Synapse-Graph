# ADR-004: Two-Tier LoRA for On-Device Adaptation

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** Specification R7, Refinement Section 4

---

## Context

SynapseGraph must adapt its models in real-time on a wearable device to track:

1. **Immediate prediction errors:** The system predicted intent X but the user intended Y. Correction must happen within microseconds.
2. **Session-level recalibration:** Neural patterns shift over a session due to fatigue, focus changes, or electrode settling.
3. **Long-term neural drift:** Over weeks to months, the user's neural encoding patterns gradually evolve due to neuroplasticity.

On-device adaptation faces three hard constraints:
- **Latency:** Real-time correction must complete in < 100 microseconds.
- **Memory:** Full model fine-tuning is infeasible on a 64 MB device.
- **Catastrophic forgetting:** Adapting to new patterns must not destroy previously learned shortcuts.

## Decision

We adopt the two-tier LoRA design from `ruvector-sona` v0.1.6 (Self-Optimizing Neural Architecture) with EWC++ (Elastic Weight Consolidation++) regularization.

### Tier Design

| Property | Fast LoRA | Slow LoRA |
|----------|-----------|-----------|
| **Rank** | 4 | 16 |
| **Target layers** | Attention heads + classification layer | GNN message-passing weights + all layers |
| **Trigger** | Prediction error or explicit feedback | Batch accumulation (idle period or N errors) |
| **Latency** | < 100 us | < 10 ms |
| **Frequency** | Per-prediction (up to 30 kHz) | Periodic (every 1-10 minutes) |
| **EWC regularization** | None (too slow) | Full EWC++ with Fisher Information Matrix |
| **Forgetting protection** | Implicit (low rank limits damage) | Explicit (EWC++ penalizes changes to important weights) |
| **WASM module** | `ruvector-learning-wasm` v2.0.4 | `ruvector-learning-wasm` v2.0.4 |
| **Persistence** | Ephemeral (reset on session end) | Persistent (survives restarts) |

### Fast LoRA Path

```
prediction_error --> compute_rank4_delta --> apply_to_attention_heads
                                        --> apply_to_classification_layer
                                        --> log_to_dag
```

The fast path is designed for immediate course correction. It modifies only the attention heads (`ruvector-attention`) and the final intent classification layer. The rank-4 constraint limits the expressiveness of each update, which serves as implicit regularization: no single fast update can dramatically alter the model's behavior.

Fast LoRA deltas are **ephemeral** -- they accumulate during a session but are not persisted. If they prove useful across sessions, they are absorbed into the slow LoRA during consolidation.

### Slow LoRA Path

```
accumulated_errors --> compute_fisher_information_matrix
                  --> compute_rank16_delta_with_ewc++
                  --> apply_to_all_layers (GNN, attention, classification)
                  --> persist_delta_to_storage
                  --> log_to_dag
```

The slow path runs during idle periods (e.g., between user interactions, during charging). It computes the Fisher Information Matrix to identify which weights are important for previously learned shortcuts, then applies rank-16 LoRA updates regularized by EWC++ to avoid changing those weights.

Slow LoRA deltas are **persistent** -- they are saved to the device and restored on restart.

### Delta Propagation

After either LoRA path executes, the resulting delta is propagated to all affected modules:

```
sona_engine.last_delta() --> gnn_engine.apply_lora_delta()
                        --> attention.apply_lora_delta()
```

### Crate Mapping

- `ruvector-sona` v0.1.6: Core two-tier LoRA engine with EWC++ implementation
- `ruvector-learning-wasm` v2.0.4: WASM bindings for on-device LoRA execution
- `ruvector-attention` v2.0.4: Attention layers receiving fast LoRA deltas
- `ruvector-gnn` v2.0.5: GNN layers receiving slow LoRA deltas
- `ruvector-math` v2.0.4: Fisher Information Matrix computation for EWC++

## Consequences

### Positive

- < 100 us fast adaptation enables real-time correction at neural recording rates.
- EWC++ on the slow path prevents catastrophic forgetting of established shortcuts.
- The two-tier design naturally maps to the temporal hierarchy: fast corrections for immediate intent, slow consolidation for long-term learning.
- LoRA's low-rank updates are memory-efficient: rank-4 adds < 1 KB per layer, rank-16 adds < 16 KB per layer.

### Negative

- Two separate LoRA paths increase implementation complexity.
- The fast path lacks explicit forgetting protection; a sustained burst of errors could temporarily degrade the model.
- EWC++ requires computing and storing the Fisher Information Matrix, which adds memory overhead (proportional to number of model parameters).

### Risks

- **Fast LoRA oscillation:** If the BCI produces conflicting predictions rapidly, fast LoRA could oscillate. Mitigation: apply exponential moving average to fast LoRA deltas with a damping factor.
- **Fisher matrix staleness:** If the slow LoRA's Fisher matrix becomes outdated (user's cognitive patterns have fundamentally changed), EWC++ may over-protect obsolete weights. Mitigation: periodically recompute the Fisher matrix from the current graph state; expose a "full reset" API for clinical use.
- **WASM performance:** LoRA matrix multiplications in WASM may be slower than native. Mitigation: `ruvector-learning-wasm` uses SIMD intrinsics where available; the rank-4 fast path is small enough to fit in L1 cache.

## Alternatives Considered

1. **Single LoRA rank:** A single rank cannot simultaneously satisfy < 100 us latency (requires low rank) and sufficient expressiveness for long-term learning (requires higher rank).
2. **Full fine-tuning with gradient accumulation:** Memory-infeasible on a 64 MB wearable device.
3. **Adapter modules (prefix tuning, prompt tuning):** These are designed for language models and don't map well to GNN + attention architectures.
4. **Online SGD without LoRA:** Unstructured gradient updates risk catastrophic forgetting without the implicit regularization that low-rank structure provides.
