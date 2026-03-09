//! GraphHealer — self-healing via min-cut analysis (DDD-004, ADR-005).
//!
//! Identifies weak points in the cognitive graph topology using approximate
//! min-cut analysis (conceptually backed by ruvector-mincut). Heals by
//! reinforcing bottlenecks with historical evidence or pruning spurious bridges.

use synapse_graph_types::{
    CognitiveEdge, EdgeType, HealingResult, NodeId, TemporalSequence, Timestamp, WeakPoint,
};

use crate::hypergraph::HypergraphStore;

/// Configuration for the graph healer.
#[derive(Clone, Debug)]
pub struct HealerConfig {
    /// Minimum min-cut value below which a connection is considered weak.
    /// Default: 2.0.
    pub min_cut_threshold: f32,
    /// Maximum number of reinforcement edges to add per weak point. Default: 3.
    pub max_reinforcement_edges: usize,
    /// Microseconds since last activation before a node is considered dead.
    /// Default: 3_600_000_000 (1 hour).
    pub prune_age_us: u64,
    /// Current timestamp for age calculations.
    pub current_time: Timestamp,
}

impl Default for HealerConfig {
    fn default() -> Self {
        Self {
            min_cut_threshold: 2.0,
            max_reinforcement_edges: 3,
            prune_age_us: 3_600_000_000,
            current_time: Timestamp(0),
        }
    }
}

/// The graph healer — identifies and repairs weak topology.
pub struct GraphHealer {
    config: HealerConfig,
}

impl GraphHealer {
    pub fn new(config: HealerConfig) -> Self {
        Self { config }
    }

    /// Update the current time used for age calculations.
    pub fn set_current_time(&mut self, time: Timestamp) {
        self.config.current_time = time;
    }

    /// Analyze the graph for weak points using approximate min-cut.
    ///
    /// For each pair of connected components or loosely connected subgraphs,
    /// identifies bottleneck edges with total weight below threshold.
    pub fn analyze(&self, graph: &HypergraphStore) -> Vec<WeakPoint> {
        let mut weak_points = Vec::new();
        let edges = graph.edges();

        if edges.is_empty() || graph.node_count() < 2 {
            return weak_points;
        }

        // Approximate min-cut: for each edge, check if removing it would
        // increase the number of connected components. This is a simplified
        // bridge-detection approach.
        //
        // In production this would use ruvector-mincut's Stoer-Wagner algorithm.
        // Here we detect bridge edges: edges whose removal disconnects the graph.

        let nodes: Vec<NodeId> = graph.nodes().iter().map(|n| n.id).collect();

        for edge in &edges {
            // Compute connectivity without this edge.
            let cut_value = self.estimate_cut_value(graph, edge);

            if cut_value < self.config.min_cut_threshold {
                // This is a bottleneck. Partition nodes into two regions
                // based on which side of the bridge they're on.
                let (region_a, region_b) =
                    self.partition_around_edge(graph, edge, &nodes);

                if !region_a.is_empty() && !region_b.is_empty() {
                    weak_points.push(WeakPoint {
                        region_a,
                        region_b,
                        min_cut_value: cut_value,
                        bridge_edges: vec![(edge.source, edge.target)],
                    });
                }
            }
        }

        // Deduplicate: if the same two regions appear multiple times, keep the
        // one with the lowest cut value.
        weak_points.sort_by(|a, b| {
            a.min_cut_value
                .partial_cmp(&b.min_cut_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        weak_points.dedup_by(|a, b| {
            // Consider two weak points the same if they share bridge edges.
            a.bridge_edges == b.bridge_edges
        });

        weak_points
    }

    /// Heal the graph by reinforcing weak points or pruning spurious bridges.
    pub fn heal(
        &self,
        graph: &mut HypergraphStore,
        weak_points: Vec<WeakPoint>,
        historical_sequences: &[TemporalSequence],
    ) -> HealingResult {
        let mut weak_points_reinforced = 0usize;
        let mut edges_added = 0usize;
        let mut edges_removed = 0usize;

        for wp in &weak_points {
            // Search historical sequences for evidence connecting region_a and region_b.
            let evidence = self.find_evidence(&wp.region_a, &wp.region_b, graph, historical_sequences);

            if !evidence.is_empty() {
                // Reinforce with evidence-based edges.
                let mut added = 0;
                for (source, target, weight) in evidence {
                    if added >= self.config.max_reinforcement_edges {
                        break;
                    }
                    graph.upsert_edge(source, target, EdgeType::Reinforcement, weight);
                    added += 1;
                    edges_added += 1;
                }
                weak_points_reinforced += 1;
            } else {
                // No evidence: prune the spurious bridge edges.
                for &(src, tgt) in &wp.bridge_edges {
                    let removed = graph.remove_edges_where(|e| {
                        e.source == src && e.target == tgt
                            || e.source == tgt && e.target == src
                    });
                    edges_removed += removed;
                }
            }
        }

        // Prune dead nodes.
        let dead_node_ids: Vec<NodeId> = graph
            .nodes()
            .iter()
            .filter(|n| {
                self.config.current_time.0 > n.last_activated.0
                    && self.config.current_time.0 - n.last_activated.0 > self.config.prune_age_us
            })
            .map(|n| n.id)
            .collect();

        let dead_regions_pruned = if dead_node_ids.is_empty() { 0 } else { 1 };
        let nodes_removed = dead_node_ids.len();

        for id in &dead_node_ids {
            // Count edges that will be removed with the node.
            let incident = graph.neighbors(*id).len();
            edges_removed += incident;
            graph.remove_node(*id);
        }

        HealingResult {
            weak_points_reinforced,
            dead_regions_pruned,
            nodes_removed,
            edges_added,
            edges_removed,
        }
    }

    /// Estimate the cut value around an edge.
    ///
    /// Returns the total weight of edges crossing the partition created by
    /// conceptually removing this edge. Low values indicate a bottleneck.
    fn estimate_cut_value(&self, graph: &HypergraphStore, edge: &CognitiveEdge) -> f32 {
        // Count the number of distinct paths between source and target
        // excluding this edge. This is a simplified connectivity metric.
        let source_neighbors: Vec<NodeId> = graph
            .neighbors(edge.source)
            .iter()
            .filter(|(id, _)| *id != edge.target)
            .map(|(id, _)| *id)
            .collect();

        let target_neighbors: Vec<NodeId> = graph
            .neighbors(edge.target)
            .iter()
            .filter(|(id, _)| *id != edge.source)
            .map(|(id, _)| *id)
            .collect();

        // Cut value: number of alternative paths (shared neighbors provide
        // alternative routes, each contributing weight).
        let mut cut_value = 0.0f32;
        for sn in &source_neighbors {
            if target_neighbors.contains(sn) {
                // Shared neighbor provides an alternative path.
                cut_value += 1.0;
            }
        }

        // Add the weight of the edge itself to the cut value.
        // A standalone bridge has cut_value = edge.weight only.
        cut_value + edge.weight
    }

    /// Partition nodes into two regions around an edge.
    fn partition_around_edge(
        &self,
        graph: &HypergraphStore,
        edge: &CognitiveEdge,
        all_nodes: &[NodeId],
    ) -> (Vec<NodeId>, Vec<NodeId>) {
        // BFS from source, excluding the bridge edge, to find region_a.
        let mut region_a = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(edge.source);
        visited.insert(edge.source);

        while let Some(current) = queue.pop_front() {
            region_a.push(current);
            for (nbr, e) in graph.neighbors(current) {
                if visited.contains(&nbr) {
                    continue;
                }
                // Skip the bridge edge.
                if (e.source == edge.source && e.target == edge.target)
                    || (e.source == edge.target && e.target == edge.source)
                {
                    continue;
                }
                visited.insert(nbr);
                queue.push_back(nbr);
            }
        }

        let region_b: Vec<NodeId> = all_nodes
            .iter()
            .filter(|id| !visited.contains(id))
            .copied()
            .collect();

        (region_a, region_b)
    }

    /// Find temporal sequence evidence connecting two regions.
    fn find_evidence(
        &self,
        region_a: &[NodeId],
        region_b: &[NodeId],
        graph: &HypergraphStore,
        sequences: &[TemporalSequence],
    ) -> Vec<(NodeId, NodeId, f32)> {
        let mut evidence = Vec::new();

        for seq in sequences {
            // Map cluster IDs to node IDs.
            let from_node = graph.find_node_by_cluster(seq.from_cluster);
            let to_node = graph.find_node_by_cluster(seq.to_cluster);

            if let (Some(from), Some(to)) = (from_node, to_node) {
                let a_has_from = region_a.contains(&from);
                let a_has_to = region_a.contains(&to);
                let b_has_from = region_b.contains(&from);
                let b_has_to = region_b.contains(&to);

                // Evidence: sequence crosses from one region to the other.
                if (a_has_from && b_has_to) || (b_has_from && a_has_to) {
                    evidence.push((from, to, seq.confidence));
                }
            }
        }

        evidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::ClusterMetadata;
    use synapse_graph_types::ClusterId;

    fn make_metadata(cluster_id: u64, time: u64) -> ClusterMetadata {
        ClusterMetadata {
            source_cluster: ClusterId(cluster_id),
            member_count: 10,
            temporal_span: (Timestamp(time), Timestamp(time)),
        }
    }

    #[test]
    fn analyze_detects_bottleneck() {
        let mut graph = HypergraphStore::new();

        // Create two subgraphs connected by a single edge (bottleneck).
        // Subgraph A: nodes 1, 2, 3 (fully connected).
        let a1 = graph.upsert_node(vec![1.0, 0.0, 0.0], make_metadata(1, 1000));
        let a2 = graph.upsert_node(vec![0.0, 1.0, 0.0], make_metadata(2, 1000));
        let a3 = graph.upsert_node(vec![-1.0, 0.0, 0.0], make_metadata(3, 1000));
        graph.upsert_edge(a1, a2, EdgeType::CoActivation, 0.9);
        graph.upsert_edge(a2, a3, EdgeType::CoActivation, 0.9);
        graph.upsert_edge(a1, a3, EdgeType::CoActivation, 0.9);

        // Subgraph B: nodes 4, 5, 6 (fully connected).
        let b1 = graph.upsert_node(vec![0.0, 0.0, 1.0], make_metadata(4, 1000));
        let b2 = graph.upsert_node(vec![0.0, 0.0, -1.0], make_metadata(5, 1000));
        let b3 = graph.upsert_node(vec![0.0, -1.0, 0.0], make_metadata(6, 1000));
        graph.upsert_edge(b1, b2, EdgeType::CoActivation, 0.9);
        graph.upsert_edge(b2, b3, EdgeType::CoActivation, 0.9);
        graph.upsert_edge(b1, b3, EdgeType::CoActivation, 0.9);

        // Single bridge edge between the subgraphs.
        graph.upsert_edge(a3, b1, EdgeType::CoActivation, 0.3);

        let healer = GraphHealer::new(HealerConfig {
            min_cut_threshold: 2.0,
            ..Default::default()
        });

        let weak_points = healer.analyze(&graph);
        assert!(
            !weak_points.is_empty(),
            "should detect the bottleneck bridge"
        );

        // The bridge edge should be identified.
        let bridge_found = weak_points.iter().any(|wp| {
            wp.bridge_edges
                .iter()
                .any(|&(s, t)| (s == a3 && t == b1) || (s == b1 && t == a3))
        });
        assert!(bridge_found, "bridge edge a3-b1 should be in weak points");
    }

    #[test]
    fn heal_reinforces_with_evidence() {
        let mut graph = HypergraphStore::new();

        // Two nodes connected by a single weak edge.
        let n1 = graph.upsert_node(vec![1.0, 0.0], make_metadata(1, 1000));
        let n2 = graph.upsert_node(vec![0.0, 1.0], make_metadata(2, 1000));
        graph.upsert_edge(n1, n2, EdgeType::CoActivation, 0.3);

        let weak_points = vec![WeakPoint {
            region_a: vec![n1],
            region_b: vec![n2],
            min_cut_value: 0.3,
            bridge_edges: vec![(n1, n2)],
        }];

        // Historical evidence connecting clusters 1 and 2.
        let sequences = vec![TemporalSequence {
            from_cluster: ClusterId(1),
            to_cluster: ClusterId(2),
            confidence: 0.8,
            observed_count: 5,
            average_delay_us: 1000,
        }];

        let healer = GraphHealer::new(HealerConfig::default());
        let result = healer.heal(&mut graph, weak_points, &sequences);

        assert_eq!(result.weak_points_reinforced, 1);
        assert!(result.edges_added > 0);

        // Should have a Reinforcement edge now.
        let has_reinforcement = graph
            .edges()
            .iter()
            .any(|e| e.edge_type == EdgeType::Reinforcement);
        assert!(has_reinforcement, "should have added reinforcement edge");
    }

    #[test]
    fn heal_prunes_dead_nodes() {
        let mut graph = HypergraphStore::new();

        // One old node and one recent node.
        let _old = graph.upsert_node(vec![1.0, 0.0], make_metadata(1, 100));
        let _new = graph.upsert_node(vec![0.0, 1.0], make_metadata(2, 5_000_000_000));

        let healer = GraphHealer::new(HealerConfig {
            prune_age_us: 3_600_000_000, // 1 hour
            current_time: Timestamp(5_000_000_000),
            ..Default::default()
        });

        let result = healer.heal(&mut graph, Vec::new(), &[]);

        assert!(result.nodes_removed > 0, "old node should be pruned");
        assert_eq!(graph.node_count(), 1, "only the recent node should remain");
    }

    #[test]
    fn heal_prunes_spurious_bridge_without_evidence() {
        let mut graph = HypergraphStore::new();

        let n1 = graph.upsert_node(vec![1.0, 0.0], make_metadata(1, 1000));
        let n2 = graph.upsert_node(vec![0.0, 1.0], make_metadata(2, 1000));
        graph.upsert_edge(n1, n2, EdgeType::CoActivation, 0.3);

        let weak_points = vec![WeakPoint {
            region_a: vec![n1],
            region_b: vec![n2],
            min_cut_value: 0.3,
            bridge_edges: vec![(n1, n2)],
        }];

        // No historical evidence.
        let healer = GraphHealer::new(HealerConfig::default());
        let result = healer.heal(&mut graph, weak_points, &[]);

        assert!(result.edges_removed > 0, "bridge should be pruned");
        assert_eq!(graph.edge_count(), 0);
    }
}
