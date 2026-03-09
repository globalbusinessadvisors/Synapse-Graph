//! VectorMemory aggregate root (DDD-002).
//!
//! The single entry point for embedding storage and retrieval. Accepts only
//! [`GatedEmbedding`] values (compile-time enforcement from ADR-002).

use std::collections::HashMap;

use synapse_graph_types::{
    EmbeddingCluster, EmbeddingId, EmbeddingStored, GatedEmbedding, InsertError,
    SearchResult, TemporalMeta, TemporalNeighbor, Timestamp,
};

use crate::collections::CollectionManager;
use crate::hnsw::{HnswConfig, HnswStore};

/// Configuration for the VectorMemory aggregate.
#[derive(Clone, Debug)]
pub struct VectorMemoryConfig {
    pub hnsw: HnswConfig,
    pub dimension: usize,
}

impl Default for VectorMemoryConfig {
    fn default() -> Self {
        Self {
            hnsw: HnswConfig::default(),
            dimension: 128,
        }
    }
}

/// The VectorMemory aggregate root.
///
/// Invariant: only [`GatedEmbedding`] values can be inserted (compile-time).
pub struct VectorMemory {
    store: HnswStore,
    collections: CollectionManager,
    metadata: HashMap<EmbeddingId, TemporalMeta>,
    next_id: u64,
}

impl VectorMemory {
    pub fn new(config: VectorMemoryConfig) -> Self {
        Self {
            store: HnswStore::new(config.dimension, config.hnsw),
            collections: CollectionManager::new(),
            metadata: HashMap::new(),
            next_id: 1,
        }
    }

    /// Access the collection manager.
    pub fn collections(&self) -> &CollectionManager {
        &self.collections
    }

    /// Access the collection manager mutably.
    pub fn collections_mut(&mut self) -> &mut CollectionManager {
        &mut self.collections
    }

    /// Access the underlying HNSW store.
    pub fn hnsw(&self) -> &HnswStore {
        &self.store
    }

    /// Total number of stored embeddings.
    pub fn total_count(&self) -> usize {
        self.store.total_count()
    }

    fn next_id(&mut self) -> EmbeddingId {
        let id = EmbeddingId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Insert a gated embedding with its temporal metadata.
    ///
    /// Returns the assigned [`EmbeddingId`] and an [`EmbeddingStored`] event.
    pub fn insert(
        &mut self,
        embedding: GatedEmbedding,
        meta: TemporalMeta,
    ) -> Result<(EmbeddingId, EmbeddingStored), InsertError> {
        let dim = embedding.vector().len();
        if dim != self.store.dimension() {
            return Err(InsertError::DimensionMismatch {
                expected: self.store.dimension(),
                got: dim,
            });
        }

        let id = self.next_id();

        // Insert into HNSW and get neighbor list.
        let hnsw_neighbors = self.store.insert(id, embedding.vector().to_vec());

        // Assign to a collection if one matches.
        let collection_id = self.collections.find_matching(&meta);
        if let Some(cid) = collection_id {
            self.collections.increment_count(cid);
        }

        // Store metadata.
        self.metadata.insert(id, meta.clone());

        let event = EmbeddingStored {
            embedding_id: id,
            collection_id,
            metadata: meta,
            hnsw_neighbors,
        };

        Ok((id, event))
    }

    /// Search for the `k` nearest neighbors to `query`.
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.store
            .search_knn(query, k)
            .into_iter()
            .filter_map(|hit| {
                let meta = self.metadata.get(&hit.id)?.clone();
                let vector = self.store.get_vector(hit.id)?.to_vec();
                Some(SearchResult {
                    embedding_id: hit.id,
                    vector,
                    distance: hit.distance,
                    metadata: meta,
                })
            })
            .collect()
    }

    /// Group recent embeddings into clusters using HNSW neighborhood structure.
    ///
    /// The `threshold` parameter controls the maximum L2 distance for cluster
    /// membership. Embeddings whose mutual HNSW neighbors form connected
    /// components within this threshold are grouped together.
    pub fn cluster_recent(&self, threshold: f32) -> Vec<EmbeddingCluster> {
        if self.store.total_count() == 0 {
            return Vec::new();
        }

        // Union-Find over embedding IDs.
        let ids: Vec<EmbeddingId> = self.store.iter_ids().collect();
        let id_to_idx: HashMap<EmbeddingId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut parent: Vec<usize> = (0..ids.len()).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // Merge nodes that are HNSW neighbors within the threshold distance.
        for &id in &ids {
            let idx = id_to_idx[&id];
            if let Some(neighbors) = self.store.get_neighbors(id) {
                let vec_a = match self.store.get_vector(id) {
                    Some(v) => v,
                    None => continue,
                };
                for &nid in neighbors {
                    if let Some(&nidx) = id_to_idx.get(&nid) {
                        if let Some(vec_b) = self.store.get_vector(nid) {
                            let dist = l2_distance(vec_a, vec_b);
                            if dist <= threshold {
                                union(&mut parent, idx, nidx);
                            }
                        }
                    }
                }
            }
        }

        // Group by root.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..ids.len() {
            let root = find(&mut parent, i);
            groups.entry(root).or_default().push(i);
        }

        // Build clusters.
        groups
            .values()
            .map(|members| {
                let member_ids: Vec<EmbeddingId> = members.iter().map(|&i| ids[i]).collect();
                let dim = self.store.dimension();
                let mut centroid = vec![0.0f32; dim];
                let mut metas = Vec::new();
                let mut ts_min = u64::MAX;
                let mut ts_max = 0u64;

                for &mid in &member_ids {
                    if let Some(v) = self.store.get_vector(mid) {
                        for (j, val) in v.iter().enumerate() {
                            centroid[j] += val;
                        }
                    }
                    if let Some(m) = self.metadata.get(&mid) {
                        ts_min = ts_min.min(m.timestamp.0);
                        ts_max = ts_max.max(m.timestamp.0);
                        metas.push(m.clone());
                    }
                }
                let n = member_ids.len() as f32;
                for v in &mut centroid {
                    *v /= n;
                }

                EmbeddingCluster {
                    centroid,
                    member_ids,
                    metadata: metas,
                    temporal_span: (Timestamp(ts_min), Timestamp(ts_max)),
                }
            })
            .collect()
    }

    /// Find embeddings from other clusters that fired within `window_us`
    /// microseconds of the given cluster's temporal span.
    ///
    /// Returns [`TemporalNeighbor`] values with correlation scores based on
    /// temporal proximity.
    pub fn find_temporally_adjacent(
        &self,
        cluster: &EmbeddingCluster,
        window_us: u64,
    ) -> Vec<TemporalNeighbor> {
        let (span_lo, span_hi) = cluster.temporal_span;
        let search_lo = span_lo.0.saturating_sub(window_us);
        let search_hi = span_hi.0.saturating_add(window_us);
        let cluster_member_set: std::collections::HashSet<EmbeddingId> =
            cluster.member_ids.iter().copied().collect();

        let mut results = Vec::new();
        for (&id, meta) in &self.metadata {
            if cluster_member_set.contains(&id) {
                continue;
            }
            let ts = meta.timestamp.0;
            if ts >= search_lo && ts <= search_hi {
                // Correlation: inversely proportional to temporal distance from
                // the cluster's span center.
                let center = (span_lo.0 + span_hi.0) / 2;
                let temporal_dist = if ts > center {
                    ts - center
                } else {
                    center - ts
                };
                let correlation = 1.0 - (temporal_dist as f32 / window_us as f32).min(1.0);
                results.push(TemporalNeighbor {
                    embedding_id: id,
                    correlation,
                });
            }
        }
        results.sort_by(|a, b| b.correlation.partial_cmp(&a.correlation).unwrap());
        results
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{
        ChannelId, CollectionFilter, SessionId, TemporalTier,
    };

    fn make_meta(ts: u64) -> TemporalMeta {
        TemporalMeta {
            tier: TemporalTier::Fast,
            timestamp: Timestamp(ts),
            channel_id: ChannelId(0),
            session_id: SessionId(1),
            coherence_score: 0.95,
        }
    }

    fn make_gated(dim: usize, base: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..dim).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), 0.95)
    }

    #[test]
    fn insert_and_search_knn_returns_nearest() {
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: 8,
            ..Default::default()
        });

        let (id1, _) = vm.insert(make_gated(8, 1.0), make_meta(100)).unwrap();
        let (id2, _) = vm.insert(make_gated(8, 1.01), make_meta(200)).unwrap();
        let (_id3, _) = vm.insert(make_gated(8, 5.0), make_meta(300)).unwrap();

        let query: Vec<f32> = (0..8).map(|i| 1.005 + (i as f32) * 0.01).collect();
        let results = vm.search_knn(&query, 1);
        assert_eq!(results.len(), 1);
        // id1 (base 1.0) and id2 (base 1.01) are both very close to the query;
        // either is acceptable as nearest. id3 (base 5.0) should never win.
        assert!(
            results[0].embedding_id == id1 || results[0].embedding_id == id2,
            "nearest should be id1 or id2, got {:?}",
            results[0].embedding_id
        );
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: 8,
            ..Default::default()
        });
        let result = vm.insert(make_gated(4, 1.0), make_meta(0));
        assert!(matches!(result, Err(InsertError::DimensionMismatch { .. })));
    }

    #[test]
    fn embedding_stored_event() {
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: 8,
            ..Default::default()
        });
        let (id, event) = vm.insert(make_gated(8, 1.0), make_meta(100)).unwrap();
        assert_eq!(event.embedding_id, id);
        assert_eq!(event.metadata.timestamp, Timestamp(100));
    }

    #[test]
    fn cluster_recent_finds_known_clusters() {
        let dim = 8;
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: dim,
            hnsw: HnswConfig { m: 4, ..Default::default() },
            ..Default::default()
        });

        // Cluster A: base ≈ 1.0
        for i in 0..20 {
            let base = 1.0 + (i as f32) * 0.001;
            vm.insert(make_gated(dim, base), make_meta(i * 10)).unwrap();
        }
        // Cluster B: base ≈ 10.0
        for i in 0..20 {
            let base = 10.0 + (i as f32) * 0.001;
            vm.insert(make_gated(dim, base), make_meta(1000 + i * 10)).unwrap();
        }

        let clusters = vm.cluster_recent(0.5);
        // We should find at least 2 distinct clusters.
        assert!(
            clusters.len() >= 2,
            "expected at least 2 clusters, got {}",
            clusters.len()
        );
        // Each cluster should have multiple members.
        for c in &clusters {
            assert!(
                c.member_ids.len() > 1,
                "cluster has only {} members",
                c.member_ids.len()
            );
        }
    }

    #[test]
    fn find_temporally_adjacent_identifies_coactivated() {
        let dim = 8;
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: dim,
            ..Default::default()
        });

        // Insert three embeddings at times 100, 110, 500.
        let (id1, _) = vm.insert(make_gated(dim, 1.0), make_meta(100)).unwrap();
        let (id2, _) = vm.insert(make_gated(dim, 1.1), make_meta(110)).unwrap();
        let (id3, _) = vm.insert(make_gated(dim, 5.0), make_meta(500)).unwrap();

        // Build a cluster from id1 only.
        let cluster = EmbeddingCluster {
            centroid: vec![1.0; dim],
            member_ids: vec![id1],
            metadata: vec![make_meta(100)],
            temporal_span: (Timestamp(100), Timestamp(100)),
        };

        // Window of 50 µs: should find id2 (at ts=110, within 50µs) but not id3.
        let neighbors = vm.find_temporally_adjacent(&cluster, 50);
        let ids: Vec<EmbeddingId> = neighbors.iter().map(|n| n.embedding_id).collect();
        assert!(ids.contains(&id2), "id2 should be temporally adjacent");
        assert!(!ids.contains(&id3), "id3 should be outside the window");
        // Correlation for id2 should be high (close in time).
        let n2 = neighbors.iter().find(|n| n.embedding_id == id2).unwrap();
        assert!(n2.correlation > 0.5);
    }

    #[test]
    fn collections_partition_store() {
        let dim = 8;
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: dim,
            ..Default::default()
        });

        let cid = vm.collections_mut().create(
            "session-1",
            CollectionFilter {
                session_id: Some(SessionId(1)),
                channel_range: None,
                temporal_tier: None,
            },
            Timestamp(0),
        );

        let meta = TemporalMeta {
            tier: TemporalTier::Fast,
            timestamp: Timestamp(100),
            channel_id: ChannelId(0),
            session_id: SessionId(1),
            coherence_score: 0.9,
        };
        let (_, event) = vm.insert(make_gated(dim, 1.0), meta).unwrap();
        assert_eq!(event.collection_id, Some(cid));

        // Verify the collection count was incremented.
        let col = vm.collections().get(cid).unwrap();
        assert_eq!(col.embedding_count, 1);
    }

    #[test]
    fn insert_latency_benchmark() {
        let dim = 128;
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: dim,
            ..Default::default()
        });

        // Pre-populate.
        for i in 0..1000 {
            let base = (i as f32) * 0.01;
            vm.insert(make_gated(dim, base), make_meta(i)).unwrap();
        }

        let iterations = 100u32;
        let start = std::time::Instant::now();
        for i in 0..iterations as u64 {
            let base = 50.0 + (i as f32) * 0.001;
            std::hint::black_box(
                vm.insert(make_gated(dim, base), make_meta(2000 + i)).unwrap(),
            );
        }
        let per_call = start.elapsed() / iterations;
        let ceiling_us: u128 = if cfg!(debug_assertions) { 20_000 } else { 500 };
        assert!(
            per_call.as_micros() < ceiling_us,
            "insert() took {per_call:?}, expected < {ceiling_us}µs"
        );
        eprintln!("insert() dim={dim}: {per_call:?}");
    }

    #[test]
    fn search_knn_latency_benchmark() {
        let dim = 128;
        let mut vm = VectorMemory::new(VectorMemoryConfig {
            dimension: dim,
            ..Default::default()
        });

        // Populate with 1k embeddings (debug) or 10k (release).
        let count: u64 = if cfg!(debug_assertions) { 1_000 } else { 10_000 };
        for i in 0..count {
            let base = (i as f32) * 0.001;
            vm.insert(make_gated(dim, base), make_meta(i)).unwrap();
        }

        let query: Vec<f32> = (0..dim).map(|i| 5.0 + (i as f32) * 0.01).collect();
        let iterations = 50u32;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(vm.search_knn(&query, 10));
        }
        let per_call = start.elapsed() / iterations;
        // Brute-force over n embeddings. Accept generous ceiling for CI.
        let ceiling_ms: u128 = if cfg!(debug_assertions) { 500 } else { 200 };
        assert!(
            per_call.as_millis() < ceiling_ms,
            "search_knn(k=10) took {per_call:?}, expected < {ceiling_ms}ms"
        );
        eprintln!("search_knn(k=10, n=10k) dim={dim}: {per_call:?}");
    }
}
