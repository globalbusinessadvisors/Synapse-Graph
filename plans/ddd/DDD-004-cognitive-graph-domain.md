# DDD-004: Cognitive Graph Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R3, R4 | ADR-003, ADR-005
**Implementing Phase:** Phase 3

---

## Domain Overview

The Cognitive Graph domain builds and maintains the hypergraph of neural pattern relationships. Nodes represent embedding clusters (stable neural activation patterns); edges represent relationships such as co-activation, temporal sequence, and semantic similarity. The GNN operates on this graph to discover cognitive shortcuts -- subgraph motifs that predict the user's intent.

This domain also owns the self-healing graph maintenance via dynamic min-cut analysis (ADR-005).

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Cognitive Graph** | A directed, weighted hypergraph where nodes are neural pattern clusters and edges encode relationships between them. |
| **Cognitive Node** | A node representing a cluster of related neural embeddings. Contains the cluster centroid, activation metadata, and feature vector. |
| **Co-Activation Edge** | An edge between two nodes whose underlying embeddings fire within a short temporal window (< 10 ms). Weight = correlation strength. |
| **Temporal Sequence Edge** | An edge from node A to node B indicating that A's pattern consistently precedes B's pattern. Weight = confidence. |
| **Semantic Similarity Edge** | An edge between nodes whose centroids are close in the embedding space. Weight = 1 - cosine_distance. |
| **Cognitive Shortcut** | A subgraph motif (typically 3-8 nodes) with high activation frequency and strong predictive power for subsequent activations. Discovered by the GNN. |
| **Motif** | A recurring subgraph pattern. Cognitive shortcuts are motifs that predict intent. |
| **Weak Point** | A bottleneck in the graph identified by min-cut analysis. A small number of edges whose removal would disconnect large subgraphs. |
| **Graph Healing** | The process of reinforcing weak points, pruning dead regions, and rebalancing the graph density. |
| **Proof-Gated Attention** | GNN attention mechanism that produces verifiable proofs of structural consistency (ADR-003). |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 COGNITIVE GRAPH CONTEXT                            |
|                                                                   |
|  Inbound:                                                         |
|    EmbeddingCluster (from Vector Memory)                          |
|    TemporalSequence (from Temporal Learning)                      |
|                                                                   |
|  +---------------------+                                         |
|  | CognitiveGraph      |  Aggregate Root                         |
|  | (owns hypergraph +  |  Manages graph structure, GNN inference,|
|  |  GNN + healer)      |  and shortcut discovery                 |
|  +----------+----------+                                         |
|             |                                                     |
|    +--------+--------+---------+                                  |
|    |                 |         |                                   |
|    v                 v         v                                   |
|  +----------+  +----------+ +----------+                          |
|  | Hyper-   |  | GNN      | | Graph    |                          |
|  | graph    |  | Engine   | | Healer   |                          |
|  | Store    |  | (shortcut| | (mincut) |                          |
|  +----------+  |  discov.)|  +----------+                         |
|                +----------+                                       |
|                                                                   |
|  Outbound:                                                        |
|    CognitiveShortcut --> [Intent Prediction Context]              |
|    ShortcutDiscovered event --> [Provenance Context]               |
|    GraphHealed event --> [Provenance Context]                      |
|                                                                   |
+------------------------------------------------------------------+
```

### Context Map

| Relationship | Upstream | Downstream | Type |
|-------------|----------|------------|------|
| Vector Memory -> Cognitive Graph | Vector Memory (DDD-002) | Cognitive Graph | Published Language (EmbeddingCluster) |
| Temporal Learning -> Cognitive Graph | Temporal Learning (DDD-003) | Cognitive Graph | Published Language (TemporalSequence) |
| Cognitive Graph -> Intent Prediction | Cognitive Graph | Intent Prediction (DDD-004a) | Published Language (CognitiveShortcut) |
| Cognitive Graph -> Adaptation | Cognitive Graph | Adaptation (DDD-005) | Published Language (graph topology for LoRA targets) |
| Cognitive Graph -> Provenance | Cognitive Graph | Provenance (DDD-006) | Published Language (discovery + healing events) |

---

## Domain Model

### Aggregates

#### CognitiveGraph (Aggregate Root)

```rust
/// Aggregate root for the cognitive graph and all graph intelligence operations.
/// Invariant: the graph is always in a self-healed state (no critical weak points).
/// Invariant: all shortcuts are discovered via proof-gated attention (ADR-003).
struct CognitiveGraph {
    hypergraph: HypergraphStore,        // ruvector-graph
    gnn_engine: GnnEngine,              // ruvector-gnn + ruvector-graph-transformer
    healer: GraphHealer,                // ruvector-mincut
    shortcuts: ShortcutRegistry,
    config: CognitiveGraphConfig,
}

impl CognitiveGraph {
    /// Update the graph with new clusters and sequences.
    fn update(&mut self, clusters: Vec<EmbeddingCluster>,
              sequences: Vec<TemporalSequence>) -> GraphUpdateResult;

    /// Discover cognitive shortcuts via GNN inference.
    fn discover_shortcuts(&mut self) -> Vec<CognitiveShortcut>;

    /// Run self-healing graph maintenance.
    fn heal(&mut self) -> HealingResult;

    /// Export graph topology for GNN inference.
    fn to_adjacency(&self) -> AdjacencyMatrix;
    fn node_features(&self) -> FeatureMatrix;
}
```

### Entities

#### CognitiveNode

```rust
/// A node in the cognitive graph representing a neural pattern cluster.
/// Identity: node_id (derived from cluster centroid hash).
struct CognitiveNode {
    id: NodeId,
    centroid: Vec<f32>,
    feature_vector: Vec<f32>,       // GNN-computed node features
    activation_count: u64,
    last_activated: Timestamp,
    first_seen: Timestamp,
    source_cluster: ClusterId,
}
```

#### CognitiveEdge

```rust
/// An edge in the cognitive graph.
/// Identity: (source_id, target_id, edge_type).
struct CognitiveEdge {
    source: NodeId,
    target: NodeId,
    edge_type: EdgeType,
    weight: f32,
    created_at: Timestamp,
    last_reinforced: Timestamp,
    reinforcement_count: u32,
}

enum EdgeType {
    CoActivation,       // temporal proximity < 10ms
    TemporalSequence,   // consistent ordering A -> B
    SemanticSimilarity, // embedding space proximity
    Reinforcement,      // added by graph healer
}
```

#### CognitiveShortcut

```rust
/// A discovered subgraph motif with predictive power.
/// Identity: shortcut_id.
struct CognitiveShortcut {
    id: ShortcutId,
    nodes: Vec<NodeId>,           // 3-8 nodes
    edges: Vec<CognitiveEdge>,
    activation_frequency: f32,    // how often this motif activates
    predictive_power: f32,        // how reliably it predicts the next activation
    discovered_at: Timestamp,
    proof: AttentionProof,        // proof from proof-gated attention (ADR-003)
    intent_associations: Vec<IntentAssociation>,  // what intents this shortcut predicts
}
```

### Value Objects

#### AttentionProof

```rust
/// Verifiable proof produced by proof-gated graph attention.
/// Certifies that the GNN's attention weights are consistent
/// with the graph's structural invariants.
struct AttentionProof {
    temporal_proof: TemporalConsistencyProof,
    manifold_proof: ManifoldConsistencyProof,
    biological_proof: Option<BiologicalConsistencyProof>,
    combined_hash: [u8; 32],
}
```

#### WeakPoint

```rust
/// A bottleneck in the graph identified by min-cut analysis.
struct WeakPoint {
    region_a: Vec<NodeId>,
    region_b: Vec<NodeId>,
    min_cut_value: f32,
    bridge_edges: Vec<CognitiveEdge>,
}
```

### Domain Events

```rust
/// Published when new cognitive shortcuts are discovered.
struct ShortcutDiscovered {
    shortcut: CognitiveShortcut,
    proof: AttentionProof,
    discovery_latency_ms: f32,
}

/// Published when the graph healer modifies the graph.
struct GraphHealed {
    weak_points_reinforced: u32,
    dead_regions_pruned: u32,
    nodes_removed: u32,
    edges_added: u32,
    edges_removed: u32,
}

/// Published when a proof-gated attention check fails.
struct ProofFailure {
    module: AttentionModule,     // Temporal, Manifold, or Biological
    affected_subgraph: Vec<NodeId>,
    failure_reason: String,
}

/// Published when the graph structure changes significantly.
struct GraphTopologyChanged {
    total_nodes: u64,
    total_edges: u64,
    average_min_cut: f32,
    shortcut_count: u32,
}
```

---

## Domain Services

### GnnEngine

**Backed by:** `ruvector-gnn` v2.0.5, `ruvector-graph-transformer` v2.0.4, `ruvector-gnn-wasm` v2.0.4

Runs graph neural network inference on the cognitive graph to discover shortcuts. Uses proof-gated attention with three modules (temporal, manifold, biological) as specified in ADR-003.

**Key operations:**
- `forward()`: Run GNN message-passing with proof-gated attention.
- `extract_motifs()`: Identify subgraph motifs meeting frequency and predictive power thresholds.

**Performance contract:** < 500 ms for full graph inference (batched, asynchronous).

### GraphHealer

**Backed by:** `ruvector-mincut` v2.0.4

Performs periodic self-healing operations on the cognitive graph:

1. **Min-cut analysis:** Identify weak points (bottleneck edges).
2. **Reinforce:** Add redundant edges using temporal tensor history.
3. **Prune:** Remove dead regions (nodes inactive beyond slow-tier window).
4. **Rebalance:** Merge over-dense subclusters, split sparse regions.

**Performance contract:** < 100 ms total for analysis + repair.

### ShortcutRegistry

Maintains the set of active cognitive shortcuts. Provides:
- Registration of newly discovered shortcuts.
- Deregistration of shortcuts that are no longer activated.
- Lookup by partial pattern for intent prediction.

---

## Graph Construction Rules

### Node Creation

Nodes are created from `EmbeddingCluster` values produced by Vector Memory:

```
EmbeddingCluster.centroid -> CognitiveNode.centroid
EmbeddingCluster.metadata -> CognitiveNode.feature_vector
```

Existing nodes are updated (activation count incremented, last_activated refreshed) if a new cluster matches an existing node within a configurable distance threshold.

### Edge Creation

| Edge Type | Source | Rule |
|-----------|--------|------|
| CoActivation | Vector Memory `find_temporally_adjacent()` | Create edge if two clusters have embeddings within 10 ms of each other. Weight = Pearson correlation. |
| TemporalSequence | Temporal Learning `extract_sequences()` | Create edge if cluster A consistently precedes cluster B. Weight = confidence. |
| SemanticSimilarity | Vector Memory KNN | Create edge if two cluster centroids are within top-k neighbors. Weight = 1 - cosine_distance. |
| Reinforcement | GraphHealer | Create edge to strengthen weak points identified by min-cut. Weight = historical co-activation frequency. |

---

## Invariants

1. **Proof-gated shortcuts only.** Every `CognitiveShortcut` must include a valid `AttentionProof`. Shortcuts without proofs are rejected.
2. **Self-healed graph.** After every `heal()` operation, no weak point has min-cut value below the critical threshold.
3. **No orphan nodes.** Every node must have at least one edge. Isolated nodes are pruned during healing.
4. **Monotonic shortcut quality.** Shortcuts are re-validated periodically. A shortcut whose predictive power drops below threshold is deregistered.

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| CognitiveGraph | `synapse_graph::cognitive` | (application layer) |
| HypergraphStore | `synapse_graph::cognitive::hypergraph` | `ruvector-graph` v2.0.4 |
| GnnEngine | `synapse_graph::cognitive::gnn` | `ruvector-gnn` v2.0.5 |
| GnnEngine (WASM) | `synapse_graph::cognitive::gnn_wasm` | `ruvector-gnn-wasm` v2.0.4 |
| ProofGatedAttention | `synapse_graph::cognitive::attention` | `ruvector-graph-transformer` v2.0.4 |
| GraphHealer | `synapse_graph::cognitive::healer` | `ruvector-mincut` v2.0.4 |
| MotifExtraction | `synapse_graph::cognitive::motifs` | `ruvector-gnn` v2.0.5 |
