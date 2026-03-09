//! HNSW (Hierarchical Navigable Small World) index for approximate nearest
//! neighbor search (DDD-002).
//!
//! This module provides an in-process vector index that wraps the algorithmic
//! primitives specified by `ruvector-core`. The implementation uses a flat
//! brute-force layer with a navigable small-world graph overlay.

use std::collections::HashMap;

use synapse_graph_types::EmbeddingId;

/// Configuration for the HNSW index.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Number of neighbors explored during index construction.
    pub ef_construction: usize,
    /// Number of neighbors explored during search.
    pub ef_search: usize,
    /// Maximum number of connections per node per layer.
    pub m: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 50,
            m: 16,
        }
    }
}

/// Entry stored inside the HNSW graph.
#[derive(Clone, Debug)]
struct HnswNode {
    id: EmbeddingId,
    vector: Vec<f32>,
    /// Neighbor list sorted by ascending distance (truncated to `m`).
    neighbors: Vec<EmbeddingId>,
}

/// Nearest-neighbor result returned by [`HnswStore::search_knn`].
#[derive(Clone, Debug)]
pub struct KnnHit {
    pub id: EmbeddingId,
    pub distance: f32,
}

/// In-process HNSW vector index.
pub struct HnswStore {
    dimension: usize,
    config: HnswConfig,
    nodes: HashMap<EmbeddingId, HnswNode>,
    /// Insertion order for O(1) iteration.
    insertion_order: Vec<EmbeddingId>,
}

impl HnswStore {
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        Self {
            dimension,
            config,
            nodes: HashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn total_count(&self) -> usize {
        self.nodes.len()
    }

    /// Insert a vector and return the IDs of its nearest HNSW neighbors.
    pub fn insert(&mut self, id: EmbeddingId, vector: Vec<f32>) -> Vec<EmbeddingId> {
        debug_assert_eq!(vector.len(), self.dimension);

        // Find nearest neighbors from existing nodes (ef_construction candidates).
        let neighbors = self.find_nearest(&vector, self.config.ef_construction.min(self.config.m));

        let neighbor_ids: Vec<EmbeddingId> = neighbors.iter().map(|h| h.id).collect();

        // Add back-links from neighbors to the new node.
        let m = self.config.m;
        // Pre-compute distances for neighbor pruning to avoid borrow conflicts.
        let mut backlink_ops: Vec<(EmbeddingId, Option<usize>)> = Vec::new();
        for hit in &neighbors {
            if let Some(neighbor_node) = self.nodes.get(&hit.id) {
                if neighbor_node.neighbors.len() < m {
                    backlink_ops.push((hit.id, None)); // append
                } else {
                    let dist_to_new = l2_distance_sq(&neighbor_node.vector, &vector);
                    // Find the worst existing neighbor (farthest from this node).
                    let mut worst_idx = 0usize;
                    let mut worst_dist = 0.0f32;
                    for (i, &nid) in neighbor_node.neighbors.iter().enumerate() {
                        if let Some(nn) = self.nodes.get(&nid) {
                            let d = l2_distance_sq(&neighbor_node.vector, &nn.vector);
                            if d > worst_dist {
                                worst_dist = d;
                                worst_idx = i;
                            }
                        }
                    }
                    if dist_to_new < worst_dist {
                        backlink_ops.push((hit.id, Some(worst_idx)));
                    }
                }
            }
        }
        // Apply back-link mutations.
        for (nid, op) in backlink_ops {
            if let Some(neighbor_node) = self.nodes.get_mut(&nid) {
                match op {
                    None => neighbor_node.neighbors.push(id),
                    Some(idx) => neighbor_node.neighbors[idx] = id,
                }
            }
        }

        let node = HnswNode {
            id,
            vector,
            neighbors: neighbor_ids.clone(),
        };
        self.nodes.insert(id, node);
        self.insertion_order.push(id);

        neighbor_ids
    }

    /// Search for the `k` nearest neighbors to `query`.
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<KnnHit> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Two-phase search: greedy walk from a random entry point, then
        // expand to ef_search candidates. For correctness we fall back to
        // exhaustive scan when the graph is small.
        if self.nodes.len() <= self.config.ef_search * 2 {
            return self.brute_force_knn(query, k);
        }

        self.graph_search_knn(query, k)
    }

    /// Return the vector for an embedding, if stored.
    pub fn get_vector(&self, id: EmbeddingId) -> Option<&[f32]> {
        self.nodes.get(&id).map(|n| n.vector.as_slice())
    }

    /// Return the HNSW neighbor list for an embedding.
    pub fn get_neighbors(&self, id: EmbeddingId) -> Option<&[EmbeddingId]> {
        self.nodes.get(&id).map(|n| n.neighbors.as_slice())
    }

    /// Iterate over all stored IDs in insertion order.
    pub fn iter_ids(&self) -> impl Iterator<Item = EmbeddingId> + '_ {
        self.insertion_order.iter().copied()
    }

    // ── Private helpers ──────────────────────────────────────────────

    fn find_nearest(&self, query: &[f32], k: usize) -> Vec<KnnHit> {
        self.brute_force_knn(query, k)
    }

    fn brute_force_knn(&self, query: &[f32], k: usize) -> Vec<KnnHit> {
        let mut hits: Vec<KnnHit> = self
            .nodes
            .values()
            .map(|n| KnnHit {
                id: n.id,
                distance: l2_distance_sq(query, &n.vector).sqrt(),
            })
            .collect();
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        hits.truncate(k);
        hits
    }

    fn graph_search_knn(&self, query: &[f32], k: usize) -> Vec<KnnHit> {
        // Start from the most recently inserted node (a reasonable entry point).
        let entry = match self.insertion_order.last() {
            Some(&id) => id,
            None => return Vec::new(),
        };

        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut results: Vec<KnnHit> = Vec::new();

        // Seed.
        visited.insert(entry);
        if let Some(node) = self.nodes.get(&entry) {
            let dist = l2_distance_sq(query, &node.vector).sqrt();
            candidates.push(std::cmp::Reverse(OrdKnnHit { id: entry, distance: dist }));
        }

        let ef = self.config.ef_search.max(k);
        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            results.push(KnnHit {
                id: current.id,
                distance: current.distance,
            });
            if results.len() >= ef {
                break;
            }

            if let Some(node) = self.nodes.get(&current.id) {
                for &neighbor_id in &node.neighbors {
                    if visited.insert(neighbor_id) {
                        if let Some(neighbor_node) = self.nodes.get(&neighbor_id) {
                            let dist = l2_distance_sq(query, &neighbor_node.vector).sqrt();
                            candidates.push(std::cmp::Reverse(OrdKnnHit {
                                id: neighbor_id,
                                distance: dist,
                            }));
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        results
    }

}

/// Helper for ordered KNN hits in a BinaryHeap.
#[derive(Clone, Debug)]
struct OrdKnnHit {
    id: EmbeddingId,
    distance: f32,
}

impl PartialEq for OrdKnnHit {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for OrdKnnHit {}
impl PartialOrd for OrdKnnHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdKnnHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Squared L2 (Euclidean) distance.
fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, base: f32) -> Vec<f32> {
        (0..dim).map(|i| base + (i as f32) * 0.01).collect()
    }

    #[test]
    fn insert_and_search() {
        let mut store = HnswStore::new(8, HnswConfig::default());
        store.insert(EmbeddingId(1), make_vec(8, 1.0));
        store.insert(EmbeddingId(2), make_vec(8, 1.1));
        store.insert(EmbeddingId(3), make_vec(8, 5.0));

        let results = store.search_knn(&make_vec(8, 1.05), 2);
        assert_eq!(results.len(), 2);
        // Nearest should be id 1 or 2 (both very close to 1.05).
        assert!(results[0].id == EmbeddingId(1) || results[0].id == EmbeddingId(2));
        // Farthest of the two should not be id 3.
        assert_ne!(results[1].id, EmbeddingId(3));
    }

    #[test]
    fn total_count() {
        let mut store = HnswStore::new(4, HnswConfig::default());
        assert_eq!(store.total_count(), 0);
        store.insert(EmbeddingId(1), vec![0.0; 4]);
        assert_eq!(store.total_count(), 1);
    }

    #[test]
    fn neighbors_populated_on_insert() {
        let mut store = HnswStore::new(4, HnswConfig::default());
        store.insert(EmbeddingId(1), vec![1.0; 4]);
        let nbrs = store.insert(EmbeddingId(2), vec![1.1; 4]);
        assert!(!nbrs.is_empty());
        assert_eq!(nbrs[0], EmbeddingId(1));
    }
}
