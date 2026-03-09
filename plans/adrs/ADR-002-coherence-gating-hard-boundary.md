# ADR-002: Coherence Gating as a Hard Architectural Boundary

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** Specification R9, Refinement Section 2

---

## Context

BCI signals are inherently noisy. Sources of corruption include:

- **Electrode impedance changes:** Gradual degradation or sudden failure of recording electrodes.
- **Movement artifacts:** Physical motion introduces broadband noise across channels.
- **Neural recording dropouts:** Transient loss of signal producing zero or saturated readings.
- **Cross-talk:** Adjacent electrodes picking up the same neural source.

These conditions produce embeddings that are syntactically valid (correct dimensionality, finite values) but carry no cognitive content. If admitted into the HNSW vector index, they:

1. Degrade KNN search quality by polluting the neighborhood graph.
2. Introduce spurious edges in the cognitive graph.
3. Cause the GNN to discover false "shortcuts" from noisy co-activations.
4. Waste limited memory budget on wearable devices.

## Decision

We establish the `CoherenceGate` as a **hard architectural boundary** -- every embedding must pass through the gate before any downstream processing occurs. This is not a soft filter or optional preprocessing step; it is a mandatory gatekeeper enforced at the system level.

### Implementation

The gate is implemented via `cognitum-gate-kernel` v0.1.1, a `no_std` WASM module that provides:

1. **Manifold Consistency Check:** Using `ruvector-math` v2.0.4, compute the Mahalanobis distance of the incoming embedding from the running manifold estimate. Embeddings beyond a configurable threshold (default: 3 sigma) are rejected.

2. **Temporal Continuity Check:** Verify that the embedding is temporally consistent with the previous N embeddings from the same channel. Sudden discontinuities beyond the expected neural firing rate are flagged.

3. **Channel Health Check:** Track per-channel impedance and signal-to-noise ratio. If a channel falls below health thresholds, all embeddings from that channel are rejected until recovery.

4. **Verdict:** Each embedding receives a binary PERMIT or DENY verdict. DENY verdicts include a reason code for provenance logging via `ruvector-dag`.

### Performance Contract

- **Latency:** < 1 microsecond per embedding evaluation.
- **Memory:** < 256 bytes of state per channel (running statistics).
- **False positive rate (wrongly denied good embeddings):** < 0.1% under normal operating conditions.
- **False negative rate (wrongly admitted bad embeddings):** < 0.01% -- the gate errs on the side of rejection.

### Architectural Position

```
BCI Hardware --> CoherenceGate --> [rest of pipeline]
                     |
                   DENY --> ProvenanceLog (ruvector-dag)
```

The gate sits between raw BCI input and the `SpikeIngester`. No code path bypasses it. The WASM module boundary enforces this: the ingestion pipeline's public API only accepts `GatedEmbedding` structs, which can only be constructed by the coherence gate.

### Crate Dependencies

- `cognitum-gate-kernel` v0.1.1: Core gate logic, `no_std` compatible
- `ruvector-math` v2.0.4: Mahalanobis distance and manifold statistics
- `ruvector-dag` v2.0.4: Provenance logging of deny events

## Consequences

### Positive

- Guarantees data quality for all downstream processing.
- Prevents slow corruption of the cognitive graph from accumulated noise.
- The `no_std` design means the gate runs on bare-metal wearable hardware with minimal overhead.
- Type-level enforcement (`GatedEmbedding`) makes it impossible to accidentally bypass.

### Negative

- The < 1 us latency budget constrains the sophistication of the consistency check. Complex anomaly detection models are not feasible at this latency.
- False positives (rejecting valid neural data) reduce effective signal bandwidth. Under noisy conditions, this could become significant.
- Channel health tracking requires calibration data from the specific BCI device.

### Risks

- **Over-rejection under novel conditions:** A user's first-ever seizure or a medication change may produce neural patterns that look like noise to the gate. Mitigation: the slow-tier drift model can flag sustained high-rejection rates as a potential novel-but-valid condition, triggering gate threshold relaxation.
- **Gate bypass via code changes:** Mitigation: the `GatedEmbedding` type is defined in `cognitum-gate-kernel` and cannot be constructed outside the gate module. This is enforced by Rust's type system.

## Alternatives Considered

1. **Soft filtering with confidence scores:** Downstream modules would need to handle uncertain data quality, increasing complexity everywhere. A hard boundary keeps downstream code simple.
2. **Post-hoc cleaning:** Store everything, clean later. Wastes memory budget on a wearable and allows graph corruption before cleaning runs.
3. **Statistical outlier detection on the vector store:** Too slow (requires KNN search) and runs after the damage is done.
