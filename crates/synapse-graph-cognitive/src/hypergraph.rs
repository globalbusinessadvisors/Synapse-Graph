//! HypergraphStore — node/edge CRUD for the cognitive graph (DDD-004).
//!
//! Wraps the graph storage layer (conceptually ruvector-graph). Provides
//! upsert semantics for nodes and edges, adjacency/feature matrix export,
//! and neighbor queries.

use std::collections::HashMap;

use synapse_graph_types::{
    AdjacencyMatrix, ClusterMetadata, CognitiveEdge, CognitiveNode, EdgeType, FeatureMatrix,
    NodeId, Timestamp,
};

/// The hypergraph store — backing storage for the cognitive graph.
pub struct HypergraphStore {
    nodes: HashMap<NodeId, CognitiveNode>,
    /// Edges keyed by (source, target, edge_type) for upsert semantics.
    edges: HashMap<(NodeId, NodeId, EdgeTypeKey), CognitiveEdge>,
    next_node_id: u64,
}

/// Hashable wrapper for EdgeType (used as part of composite key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EdgeTypeKey(EdgeType);

impl HypergraphStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_node_id: 1,
        }
    }

    /// Upsert a node. If no node with this ID exists, create one.
    /// Returns the NodeId of the created or updated node.
    pub fn upsert_node(
        &mut self,
        centroid: Vec<f32>,
        metadata: ClusterMetadata,
    ) -> NodeId {
        let now = metadata.temporal_span.1;

        // Check if an existing node has a similar centroid (cosine > 0.95).
        if let Some(existing_id) = self.find_similar_node(&centroid, 0.95) {
            let node = self.nodes.get_mut(&existing_id).unwrap();
            node.activation_count += 1;
            node.last_activated = now;
            // Update centroid with running average.
            let n = node.activation_count as f32;
            for (i, c) in centroid.iter().enumerate() {
                if i < node.centroid.len() {
                    node.centroid[i] += (c - node.centroid[i]) / n;
                }
            }
            node.feature_vector = node.centroid.clone();
            return existing_id;
        }

        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;

        let feature_vector = centroid.clone();
        let node = CognitiveNode {
            id,
            centroid,
            feature_vector,
            activation_count: 1,
            last_activated: now,
            first_seen: metadata.temporal_span.0,
            source_cluster: metadata.source_cluster,
        };
        self.nodes.insert(id, node);
        id
    }

    /// Upsert an edge. If an edge with the same (source, target, edge_type) exists,
    /// update its weight and reinforcement count.
    pub fn upsert_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        edge_type: EdgeType,
        weight: f32,
    ) {
        let key = (source, target, EdgeTypeKey(edge_type));

        if let Some(edge) = self.edges.get_mut(&key) {
            edge.weight = (edge.weight + weight) / 2.0; // running average
            edge.reinforcement_count += 1;
            let now = self.nodes
                .values()
                .map(|n| n.last_activated)
                .max()
                .unwrap_or(Timestamp(0));
            edge.last_reinforced = now;
        } else {
            let now = self.current_timestamp();
            let edge = CognitiveEdge {
                source,
                target,
                edge_type,
                weight,
                created_at: now,
                last_reinforced: now,
                reinforcement_count: 1,
            };
            self.edges.insert(key, edge);
        }
    }

    /// Export the graph as a sparse adjacency matrix.
    pub fn to_adjacency(&self) -> AdjacencyMatrix {
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let id_to_idx: HashMap<NodeId, usize> =
            node_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let size = node_ids.len();

        let entries: Vec<(usize, usize, f32)> = self
            .edges
            .values()
            .filter_map(|e| {
                let row = id_to_idx.get(&e.source)?;
                let col = id_to_idx.get(&e.target)?;
                Some((*row, *col, e.weight))
            })
            .collect();

        AdjacencyMatrix { entries, size }
    }

    /// Export node features as a feature matrix.
    pub fn node_features(&self) -> FeatureMatrix {
        let mut features: Vec<Vec<f32>> = Vec::new();
        let mut feature_dim = 0;

        for node in self.nodes.values() {
            feature_dim = feature_dim.max(node.feature_vector.len());
            features.push(node.feature_vector.clone());
        }

        let node_count = features.len();
        FeatureMatrix {
            features,
            node_count,
            feature_dim,
        }
    }

    /// Return all nodes.
    pub fn nodes(&self) -> Vec<CognitiveNode> {
        self.nodes.values().cloned().collect()
    }

    /// Return all edges.
    pub fn edges(&self) -> Vec<CognitiveEdge> {
        self.edges.values().cloned().collect()
    }

    /// Return neighbors of a node with their connecting edges.
    pub fn neighbors(&self, node: NodeId) -> Vec<(NodeId, CognitiveEdge)> {
        self.edges
            .values()
            .filter_map(|e| {
                if e.source == node {
                    Some((e.target, e.clone()))
                } else if e.target == node {
                    Some((e.source, e.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&CognitiveNode> {
        self.nodes.get(&id)
    }

    /// Remove a node and all its incident edges.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if self.nodes.remove(&id).is_some() {
            self.edges
                .retain(|_, e| e.source != id && e.target != id);
            true
        } else {
            false
        }
    }

    /// Remove edges matching a predicate.
    pub fn remove_edges_where<F: Fn(&CognitiveEdge) -> bool>(&mut self, pred: F) -> usize {
        let before = self.edges.len();
        self.edges.retain(|_, e| !pred(e));
        before - self.edges.len()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find a node whose centroid has cosine similarity > threshold with the given vector.
    fn find_similar_node(&self, centroid: &[f32], threshold: f32) -> Option<NodeId> {
        for node in self.nodes.values() {
            if cosine_similarity(&node.centroid, centroid) > threshold {
                return Some(node.id);
            }
        }
        None
    }

    /// Find the node ID whose source_cluster matches, if any.
    pub fn find_node_by_cluster(
        &self,
        cluster_id: synapse_graph_types::ClusterId,
    ) -> Option<NodeId> {
        self.nodes
            .values()
            .find(|n| n.source_cluster == cluster_id)
            .map(|n| n.id)
    }

    fn current_timestamp(&self) -> Timestamp {
        // Use the latest node activation as an approximation.
        self.nodes
            .values()
            .map(|n| n.last_activated)
            .max()
            .unwrap_or(Timestamp(0))
    }
}

impl Default for HypergraphStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(f32::EPSILON);
    dot / denom
}

/// Compute cosine distance (1 - cosine_similarity).
pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::ClusterId;

    fn make_metadata(cluster_id: u64) -> ClusterMetadata {
        ClusterMetadata {
            source_cluster: ClusterId(cluster_id),
            member_count: 10,
            temporal_span: (Timestamp(1000), Timestamp(2000)),
        }
    }

    #[test]
    fn upsert_node_creates_new_node() {
        let mut store = HypergraphStore::new();
        let id = store.upsert_node(vec![1.0, 2.0, 3.0], make_metadata(1));
        assert_eq!(store.node_count(), 1);
        let node = store.get_node(id).unwrap();
        assert_eq!(node.activation_count, 1);
        assert_eq!(node.source_cluster, ClusterId(1));
    }

    #[test]
    fn upsert_node_merges_similar_centroid() {
        let mut store = HypergraphStore::new();
        let id1 = store.upsert_node(vec![1.0, 2.0, 3.0], make_metadata(1));
        // Very similar centroid (scaled version).
        let id2 = store.upsert_node(vec![1.001, 2.002, 3.003], make_metadata(2));
        assert_eq!(id1, id2, "similar centroids should merge");
        assert_eq!(store.node_count(), 1);
        assert_eq!(store.get_node(id1).unwrap().activation_count, 2);
    }

    #[test]
    fn upsert_edge_creates_and_updates() {
        let mut store = HypergraphStore::new();
        let n1 = store.upsert_node(vec![1.0, 0.0], make_metadata(1));
        let n2 = store.upsert_node(vec![0.0, 1.0], make_metadata(2));

        store.upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);
        assert_eq!(store.edge_count(), 1);

        // Upsert same edge type — should update, not create.
        store.upsert_edge(n1, n2, EdgeType::CoActivation, 0.6);
        assert_eq!(store.edge_count(), 1);

        let edges = store.edges();
        assert_eq!(edges[0].reinforcement_count, 2);
    }

    #[test]
    fn different_edge_types_are_separate() {
        let mut store = HypergraphStore::new();
        let n1 = store.upsert_node(vec![1.0, 0.0], make_metadata(1));
        let n2 = store.upsert_node(vec![0.0, 1.0], make_metadata(2));

        store.upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);
        store.upsert_edge(n1, n2, EdgeType::SemanticSimilarity, 0.5);
        assert_eq!(store.edge_count(), 2);
    }

    #[test]
    fn to_adjacency_correct_size() {
        let mut store = HypergraphStore::new();
        let n1 = store.upsert_node(vec![1.0, 0.0], make_metadata(1));
        let n2 = store.upsert_node(vec![0.0, 1.0], make_metadata(2));
        store.upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);

        let adj = store.to_adjacency();
        assert_eq!(adj.size, 2);
        assert_eq!(adj.entries.len(), 1);
    }

    #[test]
    fn neighbors_returns_connected_nodes() {
        let mut store = HypergraphStore::new();
        let n1 = store.upsert_node(vec![1.0, 0.0], make_metadata(1));
        let n2 = store.upsert_node(vec![0.0, 1.0], make_metadata(2));
        let n3 = store.upsert_node(vec![-1.0, 0.0], make_metadata(3));

        store.upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);
        store.upsert_edge(n1, n3, EdgeType::SemanticSimilarity, 0.5);

        let nbrs = store.neighbors(n1);
        assert_eq!(nbrs.len(), 2);
    }

    #[test]
    fn remove_node_cleans_edges() {
        let mut store = HypergraphStore::new();
        let n1 = store.upsert_node(vec![1.0, 0.0], make_metadata(1));
        let n2 = store.upsert_node(vec![0.0, 1.0], make_metadata(2));
        store.upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);

        store.remove_node(n1);
        assert_eq!(store.node_count(), 1);
        assert_eq!(store.edge_count(), 0);
    }
}
