# ADR-005: Self-Healing Cognitive Graph via Dynamic Min-Cut

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** Specification R3, Refinement Section 5

---

## Context

The cognitive graph grows continuously as new neural patterns are ingested and clustered. Over time, the graph develops structural vulnerabilities:

1. **Bottleneck edges:** A single edge connects two large subgraphs. If that edge is invalidated (e.g., the underlying embedding is evicted from the fast tier), the graph fragments.
2. **Dead regions:** Clusters of nodes that have not been activated within the slow-tier timescale window. These consume memory and add noise to GNN inference.
3. **Bridge collapse:** Temporal sequence edges that were valid during a specific session but are no longer representative of the user's current neural patterns.

Without maintenance, these vulnerabilities accumulate and degrade shortcut discovery accuracy.

## Decision

We use `ruvector-mincut` v2.0.4 to perform periodic subpolynomial dynamic min-cut analysis on the cognitive graph, combined with automated repair and pruning operations.

### Min-Cut Analysis

The min-cut algorithm identifies the minimum set of edges whose removal would disconnect the graph into components. In our context:

- **Low min-cut value** between two regions means the connection is fragile -- a few edge invalidations could fragment the graph.
- **High min-cut value** means the connection is robust -- multiple independent pathways connect the regions.

### Self-Healing Operations

| Operation | Trigger | Action | Target Latency |
|-----------|---------|--------|----------------|
| **Reinforce** | Min-cut value below threshold between important regions | Add redundant edges derived from temporal tensor history | < 100 ms per weak point |
| **Prune** | Node cluster with no activations beyond slow-tier window (default: 90 days) | Remove dead nodes and their edges; reclaim memory | < 50 ms per dead region |
| **Rebalance** | Min-cut analysis reveals asymmetric graph density | Merge overly dense subclusters; split sparse regions | < 200 ms per rebalance |

### Scheduling

Min-cut analysis runs:
- **Periodically:** Every N cognitive graph updates (default: every 1000 new nodes).
- **On demand:** When GNN shortcut discovery accuracy drops below threshold.
- **Asynchronously:** Min-cut analysis does not block the ingestion or prediction pipeline. Target: < 100 ms total for analysis + repair.

### Reinforcement Strategy

When a weak point is detected, the system:

1. Queries the temporal tensor for historical co-activation patterns between the weakly connected regions.
2. If historical evidence supports the connection, adds reinforcement edges with weight proportional to the historical frequency.
3. If no historical evidence exists, the weak connection is likely spurious and is pruned instead of reinforced.

### Crate Mapping

- `ruvector-mincut` v2.0.4: Subpolynomial dynamic min-cut algorithm
- `ruvector-graph` v2.0.4: Hypergraph substrate (target of healing operations)
- `ruvector-temporal-tensor` v2.0.4: Historical co-activation data for reinforcement decisions
- `ruvector-dag` v2.0.4: Logging of all healing operations for provenance

## Consequences

### Positive

- The cognitive graph remains well-connected and resilient to individual edge failures.
- Dead region pruning prevents unbounded memory growth.
- Historical-evidence-based reinforcement ensures that repairs are grounded in actual neural data.
- Async scheduling means healing never blocks real-time inference.

### Negative

- Min-cut computation adds periodic CPU load (< 100 ms, but not negligible on a wearable).
- Reinforcement edges add to the graph's edge count, which increases GNN computation time.
- Pruning decisions are irreversible; if a "dead" region becomes relevant again, its structure must be rediscovered from scratch.

### Risks

- **Premature pruning:** A region inactive for 90 days may become relevant again (e.g., seasonal activities, returning to a skill after a break). Mitigation: pruning preserves the centroid embedding in the slow-tier tensor, so the region can be partially reconstructed if similar patterns re-emerge.
- **Over-reinforcement:** Too many reinforcement edges can make the graph overly dense, slowing GNN inference. Mitigation: cap reinforcement edges per weak point; only reinforce if min-cut value is below a strict threshold.

## Alternatives Considered

1. **No graph maintenance:** Let the graph grow organically. This leads to fragmentation and accuracy degradation over time. Unacceptable for a long-running personal device.
2. **Periodic full graph rebuild:** Discard the graph and reconstruct from the temporal tensor. This is expensive and loses the GNN's learned edge weights.
3. **Spectral graph analysis:** More theoretically elegant than min-cut but computationally expensive for dynamic graphs in WASM. Min-cut is simpler and well-suited to the subpolynomial implementation in `ruvector-mincut`.
