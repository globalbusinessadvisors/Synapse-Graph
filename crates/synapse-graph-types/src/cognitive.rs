use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{ClusterId, IntentId, NodeId, ShortcutId, Timestamp};

/// Type of edge in the cognitive graph (DDD-004).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Two clusters co-activated within a temporal window.
    CoActivation,
    /// Ordered temporal sequence (A consistently precedes B).
    TemporalSequence,
    /// High cosine similarity between cluster centroids.
    SemanticSimilarity,
    /// Reinforcement edge added by the graph healer.
    Reinforcement,
}

/// Metadata for a cluster mapped into the cognitive graph (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterMetadata {
    pub source_cluster: ClusterId,
    pub member_count: usize,
    pub temporal_span: (Timestamp, Timestamp),
}

/// A node in the cognitive graph (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveNode {
    pub id: NodeId,
    pub centroid: Vec<f32>,
    pub feature_vector: Vec<f32>,
    pub activation_count: u64,
    pub last_activated: Timestamp,
    pub first_seen: Timestamp,
    pub source_cluster: ClusterId,
}

/// An edge in the cognitive graph (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub created_at: Timestamp,
    pub last_reinforced: Timestamp,
    pub reinforcement_count: u32,
}

// ---------------------------------------------------------------------------
// GNN engine types (DDD-004, ADR-003)
// ---------------------------------------------------------------------------

/// Proof-gated attention proof per ADR-003.
///
/// Each attention module (temporal, manifold, biological) produces a serialized
/// proof that the attention weights respect domain-specific constraints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttentionProof {
    /// Serialized proof from the temporal attention module.
    pub temporal_proof: Vec<u8>,
    /// Serialized proof from the manifold attention module.
    pub manifold_proof: Vec<u8>,
    /// Serialized proof from the biological attention module (optional).
    pub biological_proof: Option<Vec<u8>>,
    /// Combined hash of all proofs.
    pub combined_hash: [u8; 32],
}

impl AttentionProof {
    /// Verify that all proof components are consistent.
    ///
    /// In a full implementation this would cryptographically verify each proof.
    /// Here we check structural validity: non-empty proofs and hash consistency.
    pub fn verify(&self) -> bool {
        if self.temporal_proof.is_empty() || self.manifold_proof.is_empty() {
            return false;
        }

        // Verify the combined hash matches a simple checksum of all proofs.
        let computed = Self::compute_hash(
            &self.temporal_proof,
            &self.manifold_proof,
            self.biological_proof.as_deref(),
        );
        self.combined_hash == computed
    }

    /// Compute the combined hash from proof components.
    pub fn compute_hash(
        temporal: &[u8],
        manifold: &[u8],
        biological: Option<&[u8]>,
    ) -> [u8; 32] {
        // Simple deterministic hash: XOR-fold all proof bytes into 32-byte digest.
        let mut hash = [0u8; 32];
        for (i, &b) in temporal.iter().enumerate() {
            hash[i % 32] ^= b;
        }
        for (i, &b) in manifold.iter().enumerate() {
            hash[(i + 7) % 32] ^= b;
        }
        if let Some(bio) = biological {
            for (i, &b) in bio.iter().enumerate() {
                hash[(i + 13) % 32] ^= b;
            }
        }
        // Mix pass for avalanche.
        for i in 0..32 {
            hash[i] = hash[i].wrapping_add(hash[(i + 1) % 32].wrapping_mul(31));
        }
        hash
    }
}

/// Association between a shortcut and an intent (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntentAssociation {
    pub intent_id: IntentId,
    pub strength: f32,
}

/// A shortcut (motif) discovered by the GNN engine (DDD-004).
///
/// Queried by Intent Prediction in Prompt 10.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveShortcut {
    pub id: ShortcutId,
    /// Nodes forming the motif subgraph.
    pub nodes: Vec<NodeId>,
    /// Edges within the motif.
    pub edges: Vec<CognitiveEdge>,
    /// How frequently this motif activates.
    pub activation_frequency: f32,
    /// Predictive power for downstream intent prediction.
    pub predictive_power: f32,
    /// When the shortcut was discovered.
    pub discovered_at: Timestamp,
    /// Proof-gated attention proof validating this shortcut.
    pub proof: AttentionProof,
    /// Associated intents (populated by Intent Prediction, Prompt 10).
    pub intent_associations: Vec<IntentAssociation>,
}

/// Output of the GNN forward pass (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GnnOutput {
    /// Updated node feature vectors after message passing.
    pub node_embeddings: Vec<Vec<f32>>,
    /// Attention weights from the proof-gated transformer.
    pub attention_weights: AttentionWeights,
    /// One proof per attention module per subgraph region.
    pub proofs: Vec<AttentionProof>,
}

/// Attention weights from the three proof-gated modules (ADR-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttentionWeights {
    /// Temporal attention: weighted by recency.
    pub temporal: Vec<Vec<f32>>,
    /// Manifold attention: weighted by geodesic distance.
    pub manifold: Vec<Vec<f32>>,
    /// Biological attention: weighted by anatomical priors.
    pub biological: Vec<Vec<f32>>,
    /// Combined attention: 0.5 * temporal + 0.3 * manifold + 0.2 * biological.
    pub combined: Vec<Vec<f32>>,
}

/// Configuration for motif extraction.
#[derive(Clone, Debug)]
pub struct MotifConfig {
    /// Minimum motif size (nodes). Default: 3.
    pub min_size: usize,
    /// Maximum motif size (nodes). Default: 8.
    pub max_size: usize,
    /// Minimum activation frequency for a motif to qualify. Default: 0.1.
    pub min_frequency: f32,
    /// Minimum predictive power for a motif to qualify. Default: 0.3.
    pub min_predictive_power: f32,
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            min_size: 3,
            max_size: 8,
            min_frequency: 0.1,
            min_predictive_power: 0.3,
        }
    }
}

/// A weak point in the graph topology identified by min-cut analysis (DDD-004, ADR-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeakPoint {
    /// Node IDs in the first partition.
    pub region_a: Vec<NodeId>,
    /// Node IDs in the second partition.
    pub region_b: Vec<NodeId>,
    /// The min-cut value (lower = weaker connection).
    pub min_cut_value: f32,
    /// Edges forming the bridge between the two regions.
    pub bridge_edges: Vec<(NodeId, NodeId)>,
}

/// Result of a graph update cycle (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphUpdateResult {
    pub nodes_created: usize,
    pub nodes_updated: usize,
    pub edges_created: usize,
    pub edges_updated: usize,
}

/// Result of a healing cycle (DDD-004, ADR-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HealingResult {
    pub weak_points_reinforced: usize,
    pub dead_regions_pruned: usize,
    pub nodes_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
}

/// Published when the graph healer completes a healing cycle (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphHealed {
    pub weak_points_reinforced: usize,
    pub dead_regions_pruned: usize,
    pub nodes_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
}

/// Published when the graph topology changes (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphTopologyChanged {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub average_min_cut: f32,
    pub shortcut_count: usize,
}

/// Published when a shortcut (motif) is discovered by the GNN engine (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShortcutDiscovered {
    pub shortcut: CognitiveShortcut,
    pub proof: AttentionProof,
    pub discovery_latency_ms: u64,
}

/// Published when a proof-gated attention module fails verification (DDD-004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProofFailure {
    pub module: String,
    pub affected_subgraph: Vec<NodeId>,
    pub reason: String,
}

/// Adjacency matrix representation for GNN consumption.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdjacencyMatrix {
    /// Sparse representation: (row, col, weight).
    pub entries: Vec<(usize, usize, f32)>,
    /// Matrix dimension (number of nodes).
    pub size: usize,
}

/// Feature matrix for GNN node features.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureMatrix {
    /// Row-major features: features[node_index] = feature_vector.
    pub features: Vec<Vec<f32>>,
    /// Number of nodes.
    pub node_count: usize,
    /// Feature dimensionality.
    pub feature_dim: usize,
}
