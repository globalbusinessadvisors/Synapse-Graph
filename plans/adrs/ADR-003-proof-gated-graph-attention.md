# ADR-003: Proof-Gated Graph Attention for Cognitive Shortcut Discovery

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** Specification R4, Refinement Section 3

---

## Context

The GNN layer discovers cognitive shortcuts by analyzing the hypergraph of neural pattern relationships. Standard graph attention mechanisms (GAT, GATv2) compute attention weights as learned functions of node features. In a BCI context, this has two problems:

1. **No verifiability:** If the graph is corrupted (e.g., a noisy embedding bypasses the coherence gate, or a min-cut failure creates a spurious bridge), standard attention will silently propagate the corruption into shortcut predictions.

2. **Interpretability:** For a neural interface, users and clinicians need to understand *why* a particular intent was predicted. Standard attention weights are not interpretable guarantees about the relationship between input and output.

## Decision

We use `ruvector-graph-transformer` v2.0.4's proof-gated attention mechanism for all GNN inference in SynapseGraph. This crate provides 8 verified attention modules; we select three for our use case:

### Selected Attention Modules

| Module | Purpose | Proof Guarantee |
|--------|---------|-----------------|
| **Temporal** | Weights edges by recency and temporal proximity of neural events. Recent co-activations receive higher attention. | Proof that attention weights are monotonically decreasing with temporal distance; no future-leaking. |
| **Manifold** | Weights edges by geodesic distance on the neural embedding manifold. Topologically close patterns receive higher attention. | Proof that attention respects the manifold metric; no shortcut through unconnected regions. |
| **Biological** | Weights edges by known neuroanatomical connectivity priors (cortical column adjacency, white matter tract connectivity). | Proof that attention is consistent with the provided anatomical connectivity matrix. |

### How Proof-Gating Works

Each attention computation produces:
1. **Attention weights** (as usual).
2. **A verifiable proof** that the computed weights are consistent with the module's structural invariants and the input graph.

If the proof fails verification, the attention output for that subgraph is masked (set to zero), and a `ProofFailure` event is logged to the DAG. This prevents corrupted graph regions from influencing predictions.

### Module Composition

The three modules are composed via weighted sum:

```
final_attention = w_temporal * temporal_attn
               + w_manifold * manifold_attn
               + w_biological * biological_attn
```

Default weights: `w_temporal=0.5, w_manifold=0.3, w_biological=0.2`. The biological module has lowest weight because anatomical priors are approximate; the temporal module has highest weight because recency is the strongest predictor of intent relevance.

### Crate Mapping

- `ruvector-graph-transformer` v2.0.4: Proof-gated attention modules (temporal, manifold, biological)
- `ruvector-gnn` v2.0.5: GNN message-passing infrastructure that consumes attention weights
- `ruvector-gnn-wasm` v2.0.4: WASM bindings for on-device GNN inference
- `ruvector-attention` v2.0.4: Base attention primitives (geometric, graph, sparse)
- `ruvector-math` v2.0.4: Manifold geodesic computation

## Consequences

### Positive

- Corrupted graph regions are automatically silenced rather than propagating errors.
- Proof failures provide a diagnostic signal: a spike in failures indicates graph health problems.
- The biological prior module grounds shortcut discovery in neuroanatomy, reducing false positives.
- Each module's proof is independently verifiable, enabling modular testing.

### Negative

- Proof generation adds computational overhead to each GNN forward pass. Estimated 15-20% additional latency vs. standard GAT.
- Requires a neuroanatomical connectivity matrix for the biological module. This is device-specific and must be provided during configuration.
- The proof system adds complexity to the GNN codebase and its WASM bindings.

### Risks

- **Proof overhead in WASM:** The 15-20% overhead may be more significant in WASM than native. Mitigation: GNN inference is batched and asynchronous (< 500 ms target), so the overhead is absorbed within the budget.
- **Biological prior quality:** Inaccurate anatomical priors could suppress valid shortcuts. Mitigation: biological module has the lowest default weight; it can be disabled entirely via configuration.

## Alternatives Considered

1. **Standard GAT / GATv2:** Simpler but provides no guarantees about output consistency with graph structure. Silent corruption propagation is unacceptable for a neural interface.
2. **Post-hoc attention pruning:** Compute standard attention, then prune weights that violate structural constraints. This is less principled than proof-gating and may miss subtle violations.
3. **Ensemble of independent attention heads:** Multiple heads provide diversity but not verifiability. A consensus of wrong answers is still wrong.
