//! FisherInformationMatrix — diagonal Fisher approximation (DDD-006).
//!
//! Uses ruvector-math (via SonaEngine) for the diagonal Fisher approximation.
//! Tracks parameter importance for EWC++ regularization.

use synapse_graph_types::Timestamp;

use synapse_graph_cognitive::graph::CognitiveGraph;

use crate::sona::SonaEngine;

/// Diagonal Fisher information matrix for EWC++ regularization.
#[derive(Clone, Debug)]
pub struct FisherInformationMatrix {
    pub(crate) diagonal: Vec<f32>,
    pub(crate) computed_at: Timestamp,
    pub(crate) sample_count: u64,
}

impl FisherInformationMatrix {
    /// Create a new empty Fisher matrix.
    pub fn new() -> Self {
        Self {
            diagonal: Vec::new(),
            computed_at: Timestamp(0),
            sample_count: 0,
        }
    }

    /// Compute the diagonal Fisher approximation from the cognitive graph.
    ///
    /// In a full implementation, this would compute the expected squared gradient
    /// over the data distribution. Here we approximate from graph topology.
    pub fn compute(graph: &CognitiveGraph, _sona: &SonaEngine) -> Self {
        let nodes = graph.hypergraph().nodes();
        let dim = nodes
            .first()
            .map(|n| n.centroid.len())
            .unwrap_or(16);

        // Approximate Fisher diagonal from node activation statistics.
        let mut diagonal = vec![0.0f32; dim];
        let node_count = nodes.len().max(1) as f32;

        for node in &nodes {
            for (d, &v) in node.centroid.iter().enumerate() {
                if d < dim {
                    // Fisher ≈ E[grad^2] — approximate as squared centroid magnitude.
                    diagonal[d] += (v * v) / node_count;
                }
            }
        }

        // Ensure all entries are positive (importance cannot be negative).
        for v in &mut diagonal {
            *v = v.abs().max(f32::EPSILON);
        }

        Self {
            diagonal,
            computed_at: Timestamp(0),
            sample_count: nodes.len() as u64,
        }
    }

    /// Get the importance of a specific weight index.
    pub fn importance(&self, weight_index: usize) -> f32 {
        self.diagonal.get(weight_index).copied().unwrap_or(0.0)
    }

    /// Number of parameters tracked.
    pub fn param_count(&self) -> usize {
        self.diagonal.len()
    }

    /// When this matrix was computed.
    pub fn computed_at(&self) -> Timestamp {
        self.computed_at
    }

    /// Number of samples used to compute this matrix.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Return indices of the top-k most important weights.
    pub fn top_protected(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self.diagonal.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|&(i, _)| i).collect()
    }
}

impl Default for FisherInformationMatrix {
    fn default() -> Self {
        Self::new()
    }
}
