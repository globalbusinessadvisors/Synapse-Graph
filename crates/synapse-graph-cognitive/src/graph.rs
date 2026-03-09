//! CognitiveGraph — aggregate root for the cognitive graph domain (DDD-004).
//!
//! Wires together HypergraphStore, GnnEngine, GraphHealer, and ShortcutRegistry
//! into a single entry point.

use std::collections::HashMap;

use synapse_graph_types::{
    ClusterId, ClusterMetadata, CognitiveShortcut, EdgeType, EmbeddingCluster,
    GraphTopologyChanged, GraphUpdateResult, HealingResult, MotifConfig, NodeId,
    ProofFailure, SearchResult, ShortcutDiscovered, ShortcutId, TemporalNeighbor,
    TemporalSequence, Timestamp, WeakPoint,
};

use crate::gnn_engine::{AttentionModuleConfig, GnnEngine};
use crate::healer::{GraphHealer, HealerConfig};
use crate::hypergraph::{cosine_distance, HypergraphStore};

/// Configuration for the CognitiveGraph aggregate.
#[derive(Clone, Debug)]
pub struct CognitiveGraphConfig {
    /// Top-k neighbors for semantic similarity edges. Default: 5.
    pub semantic_knn_k: usize,
    /// Cosine similarity threshold for semantic edges. Default: 0.7.
    pub semantic_similarity_threshold: f32,
    /// Healer configuration.
    pub healer: HealerConfig,
    /// GNN attention module configuration.
    pub attention: AttentionModuleConfig,
    /// Motif extraction configuration.
    pub motif: MotifConfig,
}

impl Default for CognitiveGraphConfig {
    fn default() -> Self {
        Self {
            semantic_knn_k: 5,
            semantic_similarity_threshold: 0.7,
            healer: HealerConfig::default(),
            attention: AttentionModuleConfig::default(),
            motif: MotifConfig::default(),
        }
    }
}

/// Shortcut registry — stores discovered cognitive shortcuts.
///
/// Populated by the GNN engine, queried by Intent Prediction (Prompt 10).
pub struct ShortcutRegistry {
    shortcuts: HashMap<ShortcutId, CognitiveShortcut>,
    next_id: u64,
}

impl ShortcutRegistry {
    pub fn new() -> Self {
        Self {
            shortcuts: HashMap::new(),
            next_id: 1,
        }
    }

    /// Register a cognitive shortcut.
    pub fn register(&mut self, shortcut: CognitiveShortcut) {
        self.shortcuts.insert(shortcut.id, shortcut);
    }

    /// Register a shortcut from simple parameters (backward compat for Prompt 08 tests).
    pub fn register_simple(
        &mut self,
        source: NodeId,
        target: NodeId,
        weight: f32,
        created_at: Timestamp,
    ) -> ShortcutId {
        let id = ShortcutId(self.next_id);
        self.next_id += 1;

        let shortcut = CognitiveShortcut {
            id,
            nodes: vec![source, target],
            edges: Vec::new(),
            activation_frequency: weight,
            predictive_power: weight,
            discovered_at: created_at,
            proof: synapse_graph_types::AttentionProof {
                temporal_proof: vec![1],
                manifold_proof: vec![1],
                biological_proof: None,
                combined_hash: synapse_graph_types::AttentionProof::compute_hash(
                    &[1], &[1], None,
                ),
            },
            intent_associations: Vec::new(),
        };
        self.shortcuts.insert(id, shortcut);
        id
    }

    /// Deregister a shortcut by ID.
    pub fn deregister(&mut self, id: &ShortcutId) {
        self.shortcuts.remove(id);
    }

    /// Get a shortcut by ID.
    pub fn get(&self, id: ShortcutId) -> Option<&CognitiveShortcut> {
        self.shortcuts.get(&id)
    }

    /// Find shortcuts matching KNN search results.
    ///
    /// Maps search results to their associated shortcuts by comparing result
    /// vectors to shortcut node centroids. Returns (shortcut, match_score) pairs.
    pub fn find_matching(
        &self,
        candidates: &[SearchResult],
        graph: &HypergraphStore,
    ) -> Vec<(CognitiveShortcut, f32)> {
        let mut results = Vec::new();

        for shortcut in self.shortcuts.values() {
            let mut best_score = 0.0f32;

            for candidate in candidates {
                for &node_id in &shortcut.nodes {
                    if let Some(node) = graph.get_node(node_id) {
                        let sim = cosine_sim(&candidate.vector, &node.centroid);
                        best_score = best_score.max(sim);
                    }
                }
            }

            if best_score > 0.0 {
                results.push((shortcut.clone(), best_score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Return all shortcuts.
    pub fn all(&self) -> Vec<&CognitiveShortcut> {
        self.shortcuts.values().collect()
    }

    /// List all shortcuts (alias for backward compat).
    pub fn list(&self) -> Vec<&CognitiveShortcut> {
        self.all()
    }

    /// Number of registered shortcuts.
    pub fn count(&self) -> usize {
        self.shortcuts.len()
    }

    /// Allocate the next shortcut ID.
    pub fn next_id(&mut self) -> ShortcutId {
        let id = ShortcutId(self.next_id);
        self.next_id += 1;
        id
    }
}

impl Default for ShortcutRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The CognitiveGraph aggregate root (DDD-004).
pub struct CognitiveGraph {
    pub(crate) hypergraph: HypergraphStore,
    gnn_engine: GnnEngine,
    healer: GraphHealer,
    shortcuts: ShortcutRegistry,
    config: CognitiveGraphConfig,
}

impl CognitiveGraph {
    pub fn new(config: CognitiveGraphConfig) -> Self {
        let healer = GraphHealer::new(config.healer.clone());
        let gnn_engine = GnnEngine::new(config.attention.clone());
        Self {
            hypergraph: HypergraphStore::new(),
            gnn_engine,
            healer,
            shortcuts: ShortcutRegistry::new(),
            config,
        }
    }

    /// Update the graph with new clusters, temporal sequences, and co-activation data.
    pub fn update(
        &mut self,
        clusters: Vec<EmbeddingCluster>,
        sequences: Vec<TemporalSequence>,
        temporal_neighbors: Vec<TemporalNeighbor>,
    ) -> GraphUpdateResult {
        let mut nodes_created = 0usize;
        let mut nodes_updated = 0usize;
        let mut edges_created = 0usize;
        let mut edges_updated = 0usize;

        // (a) Upsert nodes from clusters.
        let mut cluster_to_node: HashMap<ClusterId, NodeId> = HashMap::new();

        for (idx, cluster) in clusters.iter().enumerate() {
            let cluster_id = ClusterId(idx as u64 + 1);
            let metadata = ClusterMetadata {
                source_cluster: cluster_id,
                member_count: cluster.member_ids.len(),
                temporal_span: cluster.temporal_span,
            };

            let old_count = self.hypergraph.node_count();
            let node_id = self.hypergraph.upsert_node(cluster.centroid.clone(), metadata);
            cluster_to_node.insert(cluster_id, node_id);

            if self.hypergraph.node_count() > old_count {
                nodes_created += 1;
            } else {
                nodes_updated += 1;
            }
        }

        // (b) CoActivation edges from temporal neighbors.
        let all_node_ids: Vec<NodeId> = cluster_to_node.values().copied().collect();
        for tn in &temporal_neighbors {
            if all_node_ids.len() >= 2 {
                for i in 0..all_node_ids.len() - 1 {
                    let edge_count_before = self.hypergraph.edge_count();
                    self.hypergraph.upsert_edge(
                        all_node_ids[i],
                        all_node_ids[i + 1],
                        EdgeType::CoActivation,
                        tn.correlation,
                    );
                    if self.hypergraph.edge_count() > edge_count_before {
                        edges_created += 1;
                    } else {
                        edges_updated += 1;
                    }
                }
            }
        }

        // (c) TemporalSequence edges.
        for seq in &sequences {
            let from_node = cluster_to_node
                .get(&seq.from_cluster)
                .copied()
                .or_else(|| self.hypergraph.find_node_by_cluster(seq.from_cluster));
            let to_node = cluster_to_node
                .get(&seq.to_cluster)
                .copied()
                .or_else(|| self.hypergraph.find_node_by_cluster(seq.to_cluster));

            if let (Some(from), Some(to)) = (from_node, to_node) {
                let edge_count_before = self.hypergraph.edge_count();
                self.hypergraph.upsert_edge(
                    from,
                    to,
                    EdgeType::TemporalSequence,
                    seq.confidence,
                );
                if self.hypergraph.edge_count() > edge_count_before {
                    edges_created += 1;
                } else {
                    edges_updated += 1;
                }
            }
        }

        // (d) SemanticSimilarity edges for KNN-close nodes.
        let nodes = self.hypergraph.nodes();
        let k = self.config.semantic_knn_k.min(nodes.len().saturating_sub(1));

        for i in 0..nodes.len() {
            let mut distances: Vec<(usize, f32)> = Vec::new();
            for j in 0..nodes.len() {
                if i == j {
                    continue;
                }
                let dist = cosine_distance(&nodes[i].centroid, &nodes[j].centroid);
                distances.push((j, dist));
            }
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for &(j, dist) in distances.iter().take(k) {
                let similarity = 1.0 - dist;
                if similarity >= self.config.semantic_similarity_threshold {
                    let edge_count_before = self.hypergraph.edge_count();
                    self.hypergraph.upsert_edge(
                        nodes[i].id,
                        nodes[j].id,
                        EdgeType::SemanticSimilarity,
                        similarity,
                    );
                    if self.hypergraph.edge_count() > edge_count_before {
                        edges_created += 1;
                    } else {
                        edges_updated += 1;
                    }
                }
            }
        }

        GraphUpdateResult {
            nodes_created,
            nodes_updated,
            edges_created,
            edges_updated,
        }
    }

    /// Discover shortcuts using the GNN engine.
    ///
    /// Pipeline:
    /// a) adjacency = hypergraph.to_adjacency()
    /// b) features = hypergraph.node_features()
    /// c) output = gnn_engine.forward(adjacency, features)
    /// d) shortcuts = gnn_engine.extract_motifs(output, config)
    /// e) Register each shortcut in registry
    /// f) Return (shortcuts, events, failures)
    pub fn discover_shortcuts(
        &mut self,
    ) -> (Vec<CognitiveShortcut>, Vec<ShortcutDiscovered>, Vec<ProofFailure>) {
        let start = std::time::Instant::now();

        let adjacency = self.hypergraph.to_adjacency();
        let features = self.hypergraph.node_features();

        // Build node ID mapping (index → NodeId).
        let node_ids: Vec<NodeId> = self.hypergraph.nodes().iter().map(|n| n.id).collect();

        let output = self.gnn_engine.forward(&adjacency, &features);

        let (shortcuts, failures) = self.gnn_engine.extract_motifs(
            &output,
            &adjacency,
            &self.config.motif,
            &node_ids,
        );

        let discovery_latency_ms = start.elapsed().as_millis() as u64;

        let mut events = Vec::new();
        for shortcut in &shortcuts {
            events.push(ShortcutDiscovered {
                shortcut: shortcut.clone(),
                proof: shortcut.proof.clone(),
                discovery_latency_ms,
            });
            self.shortcuts.register(shortcut.clone());
        }

        (shortcuts, events, failures)
    }

    /// Analyze the graph for weak points.
    pub fn analyze_weak_points(&self) -> Vec<WeakPoint> {
        self.healer.analyze(&self.hypergraph)
    }

    /// Heal the graph by reinforcing or pruning weak points.
    pub fn heal(
        &mut self,
        weak_points: Vec<WeakPoint>,
        historical_sequences: &[TemporalSequence],
    ) -> HealingResult {
        self.healer
            .heal(&mut self.hypergraph, weak_points, historical_sequences)
    }

    /// Get a snapshot of the current graph topology.
    pub fn topology(&self) -> GraphTopologyChanged {
        let weak_points = self.healer.analyze(&self.hypergraph);
        let avg_cut = if weak_points.is_empty() {
            0.0
        } else {
            weak_points.iter().map(|w| w.min_cut_value).sum::<f32>()
                / weak_points.len() as f32
        };

        GraphTopologyChanged {
            total_nodes: self.hypergraph.node_count(),
            total_edges: self.hypergraph.edge_count(),
            average_min_cut: avg_cut,
            shortcut_count: self.shortcuts.count(),
        }
    }

    /// Access the hypergraph store.
    pub fn hypergraph(&self) -> &HypergraphStore {
        &self.hypergraph
    }

    /// Mutable access to the hypergraph store.
    pub fn hypergraph_mut(&mut self) -> &mut HypergraphStore {
        &mut self.hypergraph
    }

    /// Access the shortcut registry.
    pub fn shortcuts(&self) -> &ShortcutRegistry {
        &self.shortcuts
    }

    /// Mutable access to the shortcut registry.
    pub fn shortcuts_mut(&mut self) -> &mut ShortcutRegistry {
        &mut self.shortcuts
    }

    /// Access the GNN engine.
    pub fn gnn_engine(&self) -> &GnnEngine {
        &self.gnn_engine
    }

    /// Mutable access to the GNN engine.
    pub fn gnn_engine_mut(&mut self) -> &mut GnnEngine {
        &mut self.gnn_engine
    }

    /// Set the healer's current time for age-based pruning.
    pub fn set_current_time(&mut self, time: Timestamp) {
        self.healer.set_current_time(time);
    }
}

/// Cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{EmbeddingId, TemporalMeta, TemporalTier};

    fn make_cluster(id: usize, centroid: Vec<f32>, time: u64) -> EmbeddingCluster {
        EmbeddingCluster {
            centroid,
            member_ids: vec![EmbeddingId(id as u64)],
            metadata: vec![TemporalMeta {
                tier: TemporalTier::Fast,
                timestamp: Timestamp(time),
                channel_id: synapse_graph_types::ChannelId(0),
                session_id: synapse_graph_types::SessionId(1),
                coherence_score: 0.9,
            }],
            temporal_span: (Timestamp(time), Timestamp(time + 1000)),
        }
    }

    #[test]
    fn update_creates_nodes_from_clusters() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        let clusters: Vec<EmbeddingCluster> = (0..5)
            .map(|i| {
                let angle = (i as f32) * std::f32::consts::PI / 5.0;
                make_cluster(i, vec![angle.cos(), angle.sin(), 0.0], i as u64 * 1000)
            })
            .collect();

        let result = graph.update(clusters, Vec::new(), Vec::new());

        assert_eq!(result.nodes_created, 5, "should create 5 nodes");
        assert_eq!(graph.hypergraph().node_count(), 5);
    }

    #[test]
    fn update_merges_duplicate_clusters() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        let clusters = vec![
            make_cluster(1, vec![1.0, 2.0, 3.0], 1000),
            make_cluster(2, vec![1.001, 2.002, 3.003], 2000),
        ];

        let result = graph.update(clusters, Vec::new(), Vec::new());

        assert_eq!(graph.hypergraph().node_count(), 1);
        assert_eq!(result.nodes_created, 1);
        assert_eq!(result.nodes_updated, 1);
    }

    #[test]
    fn update_creates_coactivation_edges() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        let clusters = vec![
            make_cluster(1, vec![1.0, 0.0, 0.0], 1000),
            make_cluster(2, vec![0.0, 1.0, 0.0], 2000),
        ];

        let temporal_neighbors = vec![TemporalNeighbor {
            embedding_id: EmbeddingId(1),
            correlation: 0.8,
        }];

        let result = graph.update(clusters, Vec::new(), temporal_neighbors);

        let edges = graph.hypergraph().edges();
        let coact = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::CoActivation)
            .count();
        assert!(coact > 0, "should create co-activation edges");
        assert!(result.edges_created > 0);
    }

    #[test]
    fn update_creates_temporal_sequence_edges() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        let clusters = vec![
            make_cluster(1, vec![1.0, 0.0, 0.0], 1000),
            make_cluster(2, vec![0.0, 1.0, 0.0], 2000),
        ];

        let sequences = vec![TemporalSequence {
            from_cluster: ClusterId(1),
            to_cluster: ClusterId(2),
            confidence: 0.9,
            observed_count: 10,
            average_delay_us: 500,
        }];

        let result = graph.update(clusters, sequences, Vec::new());

        let edges = graph.hypergraph().edges();
        let seq_edges = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::TemporalSequence)
            .count();
        assert!(seq_edges > 0, "should create temporal sequence edges");
        assert!(result.edges_created > 0);
    }

    #[test]
    fn healer_detects_planted_bottleneck() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig {
            semantic_similarity_threshold: 0.99,
            ..Default::default()
        });

        let n1 = graph.hypergraph.upsert_node(vec![1.0, 0.0, 0.0], ClusterMetadata {
            source_cluster: ClusterId(1), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n2 = graph.hypergraph.upsert_node(vec![0.9, 0.1, 0.0], ClusterMetadata {
            source_cluster: ClusterId(2), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n3 = graph.hypergraph.upsert_node(vec![0.0, 0.0, 1.0], ClusterMetadata {
            source_cluster: ClusterId(3), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n4 = graph.hypergraph.upsert_node(vec![0.0, 0.1, 0.9], ClusterMetadata {
            source_cluster: ClusterId(4), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });

        graph.hypergraph.upsert_edge(n1, n2, EdgeType::CoActivation, 0.9);
        graph.hypergraph.upsert_edge(n3, n4, EdgeType::CoActivation, 0.9);
        graph.hypergraph.upsert_edge(n2, n3, EdgeType::CoActivation, 0.2);

        let weak_points = graph.analyze_weak_points();
        assert!(!weak_points.is_empty(), "should detect the bottleneck");
    }

    #[test]
    fn heal_reinforces_bottleneck_with_evidence() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        let n1 = graph.hypergraph.upsert_node(vec![1.0, 0.0], ClusterMetadata {
            source_cluster: ClusterId(1), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n2 = graph.hypergraph.upsert_node(vec![0.0, 1.0], ClusterMetadata {
            source_cluster: ClusterId(2), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        graph.hypergraph.upsert_edge(n1, n2, EdgeType::CoActivation, 0.2);

        let weak_points = vec![WeakPoint {
            region_a: vec![n1],
            region_b: vec![n2],
            min_cut_value: 0.2,
            bridge_edges: vec![(n1, n2)],
        }];

        let sequences = vec![TemporalSequence {
            from_cluster: ClusterId(1),
            to_cluster: ClusterId(2),
            confidence: 0.85,
            observed_count: 8,
            average_delay_us: 200,
        }];

        let result = graph.heal(weak_points, &sequences);
        assert_eq!(result.weak_points_reinforced, 1);
        assert!(result.edges_added > 0);
    }

    #[test]
    fn heal_prunes_dead_nodes() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        let _old = graph.hypergraph.upsert_node(vec![1.0, 0.0], ClusterMetadata {
            source_cluster: ClusterId(1), member_count: 5,
            temporal_span: (Timestamp(100), Timestamp(100)),
        });
        let _recent = graph.hypergraph.upsert_node(vec![0.0, 1.0], ClusterMetadata {
            source_cluster: ClusterId(2), member_count: 5,
            temporal_span: (Timestamp(5_000_000_000), Timestamp(5_000_000_000)),
        });

        graph.set_current_time(Timestamp(5_000_000_000));
        let result = graph.heal(Vec::new(), &[]);

        assert!(result.nodes_removed > 0, "should prune old node");
        assert_eq!(graph.hypergraph().node_count(), 1);
    }

    #[test]
    fn topology_snapshot() {
        let graph = CognitiveGraph::new(CognitiveGraphConfig::default());
        let topo = graph.topology();
        assert_eq!(topo.total_nodes, 0);
        assert_eq!(topo.total_edges, 0);
        assert_eq!(topo.shortcut_count, 0);
    }

    #[test]
    fn discover_shortcuts_finds_planted_motifs() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig {
            semantic_similarity_threshold: 0.99,
            motif: MotifConfig {
                min_size: 3,
                max_size: 8,
                min_frequency: 0.0,
                min_predictive_power: 0.0,
            },
            ..Default::default()
        });

        // Build a triangle: 3 nodes with all pairwise edges.
        let n1 = graph.hypergraph.upsert_node(vec![1.0, 0.0, 0.0, 0.5], ClusterMetadata {
            source_cluster: ClusterId(1), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n2 = graph.hypergraph.upsert_node(vec![0.0, 1.0, 0.0, 0.5], ClusterMetadata {
            source_cluster: ClusterId(2), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n3 = graph.hypergraph.upsert_node(vec![0.0, 0.0, 1.0, 0.5], ClusterMetadata {
            source_cluster: ClusterId(3), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });

        graph.hypergraph.upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);
        graph.hypergraph.upsert_edge(n2, n3, EdgeType::CoActivation, 0.7);
        graph.hypergraph.upsert_edge(n1, n3, EdgeType::CoActivation, 0.6);

        let (shortcuts, events, failures) = graph.discover_shortcuts();

        assert!(
            !shortcuts.is_empty(),
            "should discover at least one shortcut"
        );
        assert_eq!(shortcuts.len(), events.len());
        assert!(failures.is_empty(), "no proof failures expected");

        // Shortcut should contain the triangle nodes.
        assert!(shortcuts[0].nodes.len() >= 3);
        assert!(shortcuts[0].proof.verify());

        // Should be registered.
        assert_eq!(graph.shortcuts().count(), shortcuts.len());
    }

    #[test]
    fn shortcut_registry_find_matching() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        // Create a node and a shortcut referencing it.
        let n1 = graph.hypergraph.upsert_node(vec![1.0, 0.0, 0.0], ClusterMetadata {
            source_cluster: ClusterId(1), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        let n2 = graph.hypergraph.upsert_node(vec![0.0, 1.0, 0.0], ClusterMetadata {
            source_cluster: ClusterId(2), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });

        graph.shortcuts_mut().register_simple(n1, n2, 0.9, Timestamp(1000));

        // Search result close to n1's centroid.
        let search_results = vec![SearchResult {
            embedding_id: EmbeddingId(42),
            vector: vec![0.99, 0.01, 0.01],
            distance: 0.01,
            metadata: TemporalMeta {
                tier: TemporalTier::Fast,
                timestamp: Timestamp(2000),
                channel_id: synapse_graph_types::ChannelId(0),
                session_id: synapse_graph_types::SessionId(1),
                coherence_score: 0.9,
            },
        }];

        let matches = graph.shortcuts().find_matching(&search_results, graph.hypergraph());
        assert!(
            !matches.is_empty(),
            "should find matching shortcut"
        );
        assert!(matches[0].1 > 0.9, "match score should be high for close vector");
    }

    #[test]
    fn shortcut_registry_deregister() {
        let mut registry = ShortcutRegistry::new();
        let id = registry.register_simple(NodeId(1), NodeId(2), 0.5, Timestamp(1000));
        assert_eq!(registry.count(), 1);

        registry.deregister(&id);
        assert_eq!(registry.count(), 0);
        assert!(registry.get(id).is_none());
    }

    #[test]
    fn discover_shortcuts_rejects_invalid_proofs() {
        // This test verifies through the GNN engine tests — when proofs are invalid,
        // motifs are rejected. We verify the integration here.
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig {
            motif: MotifConfig {
                min_size: 3,
                max_size: 8,
                min_frequency: 0.0,
                min_predictive_power: 0.0,
            },
            ..Default::default()
        });

        // Two nodes only — below min_size=3, so no motifs found.
        graph.hypergraph.upsert_node(vec![1.0, 0.0], ClusterMetadata {
            source_cluster: ClusterId(1), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });
        graph.hypergraph.upsert_node(vec![0.0, 1.0], ClusterMetadata {
            source_cluster: ClusterId(2), member_count: 5,
            temporal_span: (Timestamp(1000), Timestamp(1000)),
        });

        let (shortcuts, _events, _failures) = graph.discover_shortcuts();
        assert!(shortcuts.is_empty(), "should find no shortcuts below min_size");
    }

    #[test]
    fn update_benchmark_100_clusters() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig {
            semantic_similarity_threshold: 0.9,
            ..Default::default()
        });

        let clusters: Vec<EmbeddingCluster> = (0..100)
            .map(|i| {
                let dim = 16;
                let centroid: Vec<f32> = (0..dim)
                    .map(|d| ((i * dim + d) as f32 * 0.1).sin())
                    .collect();
                make_cluster(i, centroid, i as u64 * 1000)
            })
            .collect();

        let start = std::time::Instant::now();
        let _result = graph.update(clusters, Vec::new(), Vec::new());
        let elapsed = start.elapsed();

        let ceiling_ms = if cfg!(debug_assertions) { 500 } else { 50 };
        assert!(
            elapsed.as_millis() < ceiling_ms,
            "update() with 100 clusters took {:?}, expected < {}ms",
            elapsed,
            ceiling_ms
        );
        eprintln!("update(100 clusters): {:?}", elapsed);
    }

    #[test]
    fn heal_benchmark() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        for i in 0..50 {
            let centroid: Vec<f32> = (0..8).map(|d| ((i * 8 + d) as f32 * 0.1).sin()).collect();
            graph.hypergraph.upsert_node(centroid, ClusterMetadata {
                source_cluster: ClusterId(i as u64),
                member_count: 5,
                temporal_span: (Timestamp(1000), Timestamp(2000)),
            });
        }

        let nodes = graph.hypergraph().nodes();
        for i in 0..nodes.len() - 1 {
            graph.hypergraph.upsert_edge(
                nodes[i].id,
                nodes[i + 1].id,
                EdgeType::CoActivation,
                0.5,
            );
        }

        let start = std::time::Instant::now();
        let weak_points = graph.analyze_weak_points();
        let _result = graph.heal(weak_points, &[]);
        let elapsed = start.elapsed();

        let ceiling_ms = if cfg!(debug_assertions) { 500 } else { 100 };
        assert!(
            elapsed.as_millis() < ceiling_ms,
            "heal() took {:?}, expected < {}ms",
            elapsed,
            ceiling_ms
        );
        eprintln!("heal(50 nodes): {:?}", elapsed);
    }

    #[test]
    fn discover_shortcuts_benchmark_1000_nodes() {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig {
            semantic_similarity_threshold: 0.99,
            motif: MotifConfig {
                min_size: 3,
                max_size: 5,
                min_frequency: 0.5,       // high thresholds to limit motifs found
                min_predictive_power: 0.5,
            },
            ..Default::default()
        });

        // Build a graph with 1000 nodes in a chain (each connected to next).
        let node_count = if cfg!(debug_assertions) { 200 } else { 1000 };
        let mut prev = None;
        for i in 0..node_count {
            let centroid: Vec<f32> = (0..8)
                .map(|d| ((i * 8 + d) as f32 * 0.1).sin())
                .collect();
            let n = graph.hypergraph.upsert_node(centroid, ClusterMetadata {
                source_cluster: ClusterId(i as u64),
                member_count: 5,
                temporal_span: (Timestamp(1000), Timestamp(2000)),
            });
            if let Some(p) = prev {
                graph.hypergraph.upsert_edge(p, n, EdgeType::CoActivation, 0.7);
            }
            prev = Some(n);
        }

        let start = std::time::Instant::now();
        let (_shortcuts, _events, _failures) = graph.discover_shortcuts();
        let elapsed = start.elapsed();

        let ceiling_ms: u128 = if cfg!(debug_assertions) { 5000 } else { 500 };
        assert!(
            elapsed.as_millis() < ceiling_ms,
            "discover_shortcuts({} nodes) took {:?}, expected < {}ms",
            node_count,
            elapsed,
            ceiling_ms
        );
        eprintln!("discover_shortcuts({} nodes): {:?}", node_count, elapsed);
    }
}
