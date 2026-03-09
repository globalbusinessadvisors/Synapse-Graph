# ADR-001: Three-Tier Temporal Architecture

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** Specification R5, Refinement Section 1

---

## Context

BCI spike-train data carries information at fundamentally different timescales:

- **Millisecond-scale spike bursts** encode immediate motor intent and sensory responses.
- **Session-scale patterns** (minutes to hours) encode learning, adaptation, and task-specific neural reorganization.
- **Long-term drift** (days to months) reflects neuroplasticity, electrode impedance changes, and gradual shifts in neural encoding.

A single monolithic model attempting to learn across all three timescales faces an irreconcilable tension: fast learning destroys slow-learned structure (catastrophic forgetting), while slow learning cannot react to real-time intent.

## Decision

We decompose temporal learning into three explicitly separated tiers, each with its own data structure, update strategy, quantization level, and backing crate:

| Tier | Window | Data Structure | Update Strategy | Crate |
|------|--------|----------------|-----------------|-------|
| **Fast** | < 10 ms | Lock-free ring buffer of raw `Vec<f32>` embeddings | BTSP (Behavioral Time-Scale Plasticity) -- Hebbian-like updates on spike timing | `ruvector-nervous-system` v2.0.4 |
| **Medium** | Minutes -- Hours | Session-level compressed tensors with 8-bit quantization | Batched GNN re-training on accumulated graph deltas | `ruvector-temporal-tensor` v2.0.4 |
| **Slow** | Days -- Months | Quantized long-term drift vectors with 4-bit quantization | EWC++ regularized consolidation via SONA | `ruvector-sona` v0.1.6 |

### Tier Interaction

1. **Routing:** The `TemporalRouter` (backed by `ruvector-tiny-dancer-core` v2.0.4 FastGRNN) classifies each incoming embedding into exactly one tier based on its temporal characteristics. Classification latency target: < 10 us.

2. **Promotion:** Fast-tier patterns that persist beyond a configurable threshold (default: 500 ms of repeated activation) are promoted to the medium tier. Medium-tier patterns stable across N sessions (default: 3) are consolidated into the slow tier.

3. **Eviction:** Fast-tier ring buffer evicts oldest entries on overflow. Medium-tier tensors are re-quantized under memory pressure. Slow-tier data is never evicted but may be further quantized (4-bit -> 2-bit) if storage limits are reached.

4. **Cross-Tier Context:** The `TemporalTensor` provides a unified `get_context()` API that blends all three tiers with configurable weights (default: fast=0.5, medium=0.3, slow=0.2) for the intent prediction layer.

### Crate Mapping

- `ruvector-nervous-system` + `ruvector-nervous-system-wasm`: Fast tier BTSP processing and spike filtering
- `ruvector-temporal-tensor`: Medium tier session tensor management and cross-tier aggregation
- `ruvector-sona` + `ruvector-learning-wasm`: Slow tier EWC++ consolidation and LoRA adaptation
- `ruvector-router-core` + `ruvector-tiny-dancer-core` + `ruvector-tiny-dancer-wasm`: Temporal routing decisions
- `ruvector-math`: Drift measurement via optimal transport and information geometry

## Consequences

### Positive

- Each tier can be independently tuned, benchmarked, and tested.
- Fast-tier noise never corrupts slow-tier learned structure.
- Memory budget can be allocated per-tier based on device constraints.
- The promotion/eviction model naturally compresses information over time, matching how biological memory consolidation works.

### Negative

- Three separate codepaths increase system complexity and testing surface.
- Cross-tier promotion thresholds require empirical tuning per user / per BCI device.
- The routing decision itself adds latency (mitigated by FastGRNN's < 10 us classification).

### Risks

- **Promotion lag:** If the fast-to-medium threshold is too high, useful session patterns are lost. Mitigation: configurable thresholds with sensible defaults; telemetry on promotion rates.
- **Quantization loss:** Aggressive quantization in the slow tier may lose subtle drift signals. Mitigation: `ruvector-math` provides manifold distance metrics to validate that quantized representations preserve neighborhood structure.

## Alternatives Considered

1. **Single unified model with replay buffer:** Simpler but suffers from catastrophic forgetting. Replay buffers add memory overhead without solving the timescale mismatch.
2. **Continual learning with no tiers (EWC only):** EWC alone cannot handle the 6-order-of-magnitude timescale range (milliseconds to months). Weight importance estimates become stale.
3. **Two tiers (fast + slow):** Missing the medium tier loses session-level patterns that are critical for calibration and learning rate estimation.
