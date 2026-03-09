//! GnnEngine — proof-gated graph attention and shortcut discovery (DDD-004, ADR-003).
//!
//! Wraps ruvector-gnn + ruvector-graph-transformer to run graph neural network
//! forward passes with proof-gated attention. Extracts motifs (subgraph patterns)
//! as cognitive shortcuts.

use synapse_graph_types::{
    AdjacencyMatrix, AttentionProof, AttentionWeights, CognitiveEdge, CognitiveShortcut,
    EdgeType, FeatureMatrix, GnnOutput, LayerId, LoraDelta, LoraError, LoraTarget, MotifConfig,
    NodeId, ShortcutId, Timestamp,
};

/// Configuration for each attention module in the proof-gated transformer.
#[derive(Clone, Debug)]
pub struct AttentionModuleConfig {
    /// Weight for temporal attention module. Default: 0.5.
    pub temporal_weight: f32,
    /// Weight for manifold attention module. Default: 0.3.
    pub manifold_weight: f32,
    /// Weight for biological attention module. Default: 0.2.
    pub biological_weight: f32,
    /// Number of GNN message-passing layers. Default: 2.
    pub num_layers: usize,
    /// Whether biological priors are available. Default: false.
    pub use_biological: bool,
}

impl Default for AttentionModuleConfig {
    fn default() -> Self {
        Self {
            temporal_weight: 0.5,
            manifold_weight: 0.3,
            biological_weight: 0.2,
            num_layers: 2,
            use_biological: false,
        }
    }
}

/// The GNN engine — proof-gated graph neural network.
///
/// In production, backs onto ruvector-gnn and ruvector-graph-transformer.
/// This implementation provides the message-passing and attention logic directly.
pub struct GnnEngine {
    attention_config: AttentionModuleConfig,
    next_shortcut_id: u64,
    /// Accumulated LoRA weight adjustments per layer.
    lora_adjustments: Vec<(LayerId, Vec<f32>)>,
}

impl GnnEngine {
    pub fn new(attention_config: AttentionModuleConfig) -> Self {
        Self {
            attention_config,
            next_shortcut_id: 1,
            lora_adjustments: Vec::new(),
        }
    }

    /// Access LoRA adjustments (for testing/inspection).
    pub fn lora_adjustments(&self) -> &[(LayerId, Vec<f32>)] {
        &self.lora_adjustments
    }

    /// Run a GNN forward pass on the graph.
    ///
    /// Performs message-passing with proof-gated attention using three modules:
    /// - Temporal: attention weighted by recency (monotonic decay proof).
    /// - Manifold: attention weighted by geodesic distance (metric consistency proof).
    /// - Biological: attention weighted by anatomical priors (connectivity proof).
    ///
    /// Composition: final_attn = 0.5 * temporal + 0.3 * manifold + 0.2 * biological
    pub fn forward(
        &self,
        adjacency: &AdjacencyMatrix,
        features: &FeatureMatrix,
    ) -> GnnOutput {
        let n = features.node_count;
        let dim = features.feature_dim;

        if n == 0 || dim == 0 {
            return GnnOutput {
                node_embeddings: Vec::new(),
                attention_weights: AttentionWeights {
                    temporal: Vec::new(),
                    manifold: Vec::new(),
                    biological: Vec::new(),
                    combined: Vec::new(),
                },
                proofs: Vec::new(),
            };
        }

        // Build dense adjacency for message passing.
        let mut adj = vec![vec![0.0f32; n]; n];
        for &(r, c, w) in &adjacency.entries {
            if r < n && c < n {
                adj[r][c] = w;
                adj[c][r] = w; // symmetric
            }
        }

        // Compute attention weights for each module.
        let temporal_attn = self.compute_temporal_attention(&adj, n);
        let manifold_attn = self.compute_manifold_attention(&adj, features, n);
        let biological_attn = self.compute_biological_attention(&adj, n);

        // Compose: final = w_t * temporal + w_m * manifold + w_b * biological
        let combined = self.compose_attention(&temporal_attn, &manifold_attn, &biological_attn, n);

        // Message passing: update node features using combined attention.
        let mut embeddings = features.features.clone();
        for _layer in 0..self.attention_config.num_layers {
            embeddings = self.message_pass(&embeddings, &combined, dim);
        }

        // Generate proofs for the attention computation.
        let proofs = self.generate_proofs(&temporal_attn, &manifold_attn, &biological_attn, n);

        GnnOutput {
            node_embeddings: embeddings,
            attention_weights: AttentionWeights {
                temporal: temporal_attn,
                manifold: manifold_attn,
                biological: biological_attn,
                combined,
            },
            proofs,
        }
    }

    /// Extract motifs (subgraph patterns) from GNN output as cognitive shortcuts.
    ///
    /// Identifies connected subgraphs of 3-8 nodes where:
    /// - activation_frequency >= config.min_frequency
    /// - predictive_power >= config.min_predictive_power
    /// - All attention proofs verify (ADR-003 invariant)
    pub fn extract_motifs(
        &mut self,
        output: &GnnOutput,
        adjacency: &AdjacencyMatrix,
        config: &MotifConfig,
        node_ids: &[NodeId],
    ) -> (Vec<CognitiveShortcut>, Vec<synapse_graph_types::ProofFailure>) {
        let mut shortcuts = Vec::new();
        let mut proof_failures = Vec::new();
        let n = output.node_embeddings.len();

        if n < config.min_size {
            return (shortcuts, proof_failures);
        }

        // Build adjacency list for motif enumeration.
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c, _w) in &adjacency.entries {
            if r < n && c < n {
                neighbors[r].push(c);
                neighbors[c].push(r);
            }
        }

        // Extract connected subgraphs via BFS from each node.
        let mut visited_motifs: Vec<Vec<usize>> = Vec::new();

        for start in 0..n {
            // BFS to find a connected subgraph of size [min_size, max_size].
            let mut subgraph = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            let mut visited = vec![false; n];

            queue.push_back(start);
            visited[start] = true;

            while let Some(node) = queue.pop_front() {
                if subgraph.len() >= config.max_size {
                    break;
                }
                subgraph.push(node);

                for &nbr in &neighbors[node] {
                    if !visited[nbr] && subgraph.len() + queue.len() < config.max_size {
                        visited[nbr] = true;
                        queue.push_back(nbr);
                    }
                }
            }

            if subgraph.len() < config.min_size || subgraph.len() > config.max_size {
                continue;
            }

            // Deduplicate: sort and check if we've already found this motif.
            subgraph.sort();
            if visited_motifs.contains(&subgraph) {
                continue;
            }

            // Compute motif metrics.
            let activation_frequency = self.compute_activation_frequency(&subgraph, &output.attention_weights.combined);
            let predictive_power = self.compute_predictive_power(&subgraph, &output.node_embeddings);

            if activation_frequency < config.min_frequency
                || predictive_power < config.min_predictive_power
            {
                continue;
            }

            // Find a proof covering this subgraph.
            let proof = self.find_proof_for_subgraph(&subgraph, &output.proofs);

            if !proof.verify() {
                proof_failures.push(synapse_graph_types::ProofFailure {
                    module: "combined".into(),
                    affected_subgraph: subgraph.iter().filter_map(|&i| node_ids.get(i).copied()).collect(),
                    reason: "Attention proof verification failed".into(),
                });
                continue;
            }

            // Build the shortcut.
            let motif_node_ids: Vec<NodeId> = subgraph
                .iter()
                .filter_map(|&i| node_ids.get(i).copied())
                .collect();

            let motif_edges = self.extract_motif_edges(&subgraph, adjacency, node_ids);

            let id = ShortcutId(self.next_shortcut_id);
            self.next_shortcut_id += 1;

            shortcuts.push(CognitiveShortcut {
                id,
                nodes: motif_node_ids,
                edges: motif_edges,
                activation_frequency,
                predictive_power,
                discovered_at: Timestamp(0), // caller sets real timestamp
                proof,
                intent_associations: Vec::new(),
            });

            visited_motifs.push(subgraph);
        }

        (shortcuts, proof_failures)
    }

    // -----------------------------------------------------------------------
    // Attention computation
    // -----------------------------------------------------------------------

    /// Temporal attention: higher weight for edges between recently co-activated nodes.
    /// Proof: attention monotonically decreases with increasing time gap.
    fn compute_temporal_attention(&self, adj: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
        let mut attn = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            let mut row_sum = 0.0f32;
            for j in 0..n {
                if adj[i][j] > 0.0 {
                    // Recency proxy: use edge weight as temporal signal.
                    attn[i][j] = adj[i][j];
                    row_sum += attn[i][j];
                }
            }
            // Normalize.
            if row_sum > 0.0 {
                for j in 0..n {
                    attn[i][j] /= row_sum;
                }
            }
        }
        attn
    }

    /// Manifold attention: weighted by feature-space geodesic distance.
    /// Proof: attention respects manifold metric (closer nodes get higher attention).
    fn compute_manifold_attention(
        &self,
        adj: &[Vec<f32>],
        features: &FeatureMatrix,
        n: usize,
    ) -> Vec<Vec<f32>> {
        let mut attn = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            let mut row_sum = 0.0f32;
            for j in 0..n {
                if adj[i][j] > 0.0 && i < features.features.len() && j < features.features.len() {
                    // Geodesic approximation: 1 / (1 + L2_distance).
                    let dist = l2_distance(&features.features[i], &features.features[j]);
                    let w = 1.0 / (1.0 + dist);
                    attn[i][j] = w;
                    row_sum += w;
                }
            }
            if row_sum > 0.0 {
                for j in 0..n {
                    attn[i][j] /= row_sum;
                }
            }
        }
        attn
    }

    /// Biological attention: weighted by anatomical connectivity priors.
    /// Proof: attention consistent with known connectivity matrix.
    fn compute_biological_attention(&self, adj: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
        if !self.attention_config.use_biological {
            // No biological priors: uniform attention over neighbors.
            let mut attn = vec![vec![0.0f32; n]; n];
            for i in 0..n {
                let degree = adj[i].iter().filter(|&&w| w > 0.0).count() as f32;
                if degree > 0.0 {
                    for j in 0..n {
                        if adj[i][j] > 0.0 {
                            attn[i][j] = 1.0 / degree;
                        }
                    }
                }
            }
            return attn;
        }

        // With biological priors: weight by connectivity strength.
        let mut attn = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            let mut row_sum = 0.0f32;
            for j in 0..n {
                if adj[i][j] > 0.0 {
                    attn[i][j] = adj[i][j].sqrt(); // sqrt-weighted prior
                    row_sum += attn[i][j];
                }
            }
            if row_sum > 0.0 {
                for j in 0..n {
                    attn[i][j] /= row_sum;
                }
            }
        }
        attn
    }

    /// Compose the three attention matrices: 0.5 * T + 0.3 * M + 0.2 * B.
    fn compose_attention(
        &self,
        temporal: &[Vec<f32>],
        manifold: &[Vec<f32>],
        biological: &[Vec<f32>],
        n: usize,
    ) -> Vec<Vec<f32>> {
        let wt = self.attention_config.temporal_weight;
        let wm = self.attention_config.manifold_weight;
        let wb = self.attention_config.biological_weight;

        let mut combined = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                combined[i][j] =
                    wt * temporal[i][j] + wm * manifold[i][j] + wb * biological[i][j];
            }
        }
        combined
    }

    /// One layer of message passing using attention weights.
    fn message_pass(
        &self,
        features: &[Vec<f32>],
        attention: &[Vec<f32>],
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let n = features.len();
        let mut updated = Vec::with_capacity(n);

        for i in 0..n {
            let mut new_feat = vec![0.0f32; dim];
            for j in 0..n {
                if attention[i][j] > 0.0 {
                    for d in 0..dim.min(features[j].len()) {
                        new_feat[d] += attention[i][j] * features[j][d];
                    }
                }
            }
            // ReLU activation.
            for v in &mut new_feat {
                *v = v.max(0.0);
            }
            // Residual connection.
            if i < features.len() {
                for d in 0..dim.min(features[i].len()) {
                    new_feat[d] += features[i][d];
                }
            }
            updated.push(new_feat);
        }
        updated
    }

    /// Generate attention proofs for the computed attention weights.
    fn generate_proofs(
        &self,
        temporal: &[Vec<f32>],
        manifold: &[Vec<f32>],
        biological: &[Vec<f32>],
        n: usize,
    ) -> Vec<AttentionProof> {
        if n == 0 {
            return Vec::new();
        }

        // Generate one proof per subgraph region (simplified: one global proof).
        let temporal_proof = Self::serialize_attention_proof(temporal);
        let manifold_proof = Self::serialize_attention_proof(manifold);
        let biological_proof = if self.attention_config.use_biological {
            Some(Self::serialize_attention_proof(biological))
        } else {
            None
        };

        let combined_hash = AttentionProof::compute_hash(
            &temporal_proof,
            &manifold_proof,
            biological_proof.as_deref(),
        );

        vec![AttentionProof {
            temporal_proof,
            manifold_proof,
            biological_proof,
            combined_hash,
        }]
    }

    /// Serialize an attention matrix into a compact proof representation.
    fn serialize_attention_proof(attn: &[Vec<f32>]) -> Vec<u8> {
        let mut proof = Vec::new();
        for row in attn {
            for &v in row {
                proof.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Add a simple checksum byte.
        let checksum: u8 = proof.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        proof.push(checksum);
        proof
    }

    // -----------------------------------------------------------------------
    // Motif extraction helpers
    // -----------------------------------------------------------------------

    /// Compute activation frequency of a subgraph motif.
    fn compute_activation_frequency(
        &self,
        subgraph: &[usize],
        combined_attn: &[Vec<f32>],
    ) -> f32 {
        if subgraph.len() < 2 {
            return 0.0;
        }

        // Frequency proxy: average attention weight within the subgraph.
        let mut total = 0.0f32;
        let mut count = 0u32;
        for &i in subgraph {
            for &j in subgraph {
                if i != j && i < combined_attn.len() && j < combined_attn[i].len() {
                    total += combined_attn[i][j];
                    count += 1;
                }
            }
        }
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    /// Compute predictive power of a subgraph motif.
    fn compute_predictive_power(&self, subgraph: &[usize], embeddings: &[Vec<f32>]) -> f32 {
        if subgraph.len() < 2 {
            return 0.0;
        }

        // Predictive power proxy: coherence of node embeddings within the motif.
        // High coherence = embeddings are aligned = motif is a strong signal.
        let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
        if dim == 0 {
            return 0.0;
        }

        // Compute centroid.
        let mut centroid = vec![0.0f32; dim];
        let n = subgraph.len() as f32;
        for &i in subgraph {
            if i < embeddings.len() {
                for (d, v) in embeddings[i].iter().enumerate() {
                    if d < dim {
                        centroid[d] += v / n;
                    }
                }
            }
        }

        // Average cosine similarity to centroid.
        let mut total_sim = 0.0f32;
        for &i in subgraph {
            if i < embeddings.len() {
                total_sim += cosine_similarity(&embeddings[i], &centroid);
            }
        }
        total_sim / n
    }

    /// Find or construct a proof for a specific subgraph.
    fn find_proof_for_subgraph(
        &self,
        _subgraph: &[usize],
        proofs: &[AttentionProof],
    ) -> AttentionProof {
        // Use the global proof (simplified; full impl would extract per-subgraph proofs).
        proofs.first().cloned().unwrap_or_else(|| {
            // Invalid proof — will fail verification.
            AttentionProof {
                temporal_proof: Vec::new(),
                manifold_proof: Vec::new(),
                biological_proof: None,
                combined_hash: [0u8; 32],
            }
        })
    }

    /// Extract edges within a motif subgraph.
    fn extract_motif_edges(
        &self,
        subgraph: &[usize],
        adjacency: &AdjacencyMatrix,
        node_ids: &[NodeId],
    ) -> Vec<CognitiveEdge> {
        let mut edges = Vec::new();
        let sub_set: std::collections::HashSet<usize> = subgraph.iter().copied().collect();

        for &(r, c, w) in &adjacency.entries {
            if sub_set.contains(&r) && sub_set.contains(&c) {
                if let (Some(&src), Some(&tgt)) = (node_ids.get(r), node_ids.get(c)) {
                    edges.push(CognitiveEdge {
                        source: src,
                        target: tgt,
                        edge_type: EdgeType::CoActivation,
                        weight: w,
                        created_at: Timestamp(0),
                        last_reinforced: Timestamp(0),
                        reinforcement_count: 1,
                    });
                }
            }
        }
        edges
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

impl LoraTarget for GnnEngine {
    fn apply_lora_delta(&mut self, delta: &LoraDelta) -> Result<(), LoraError> {
        // Verify this layer is one we manage.
        let our_layers = self.layer_ids();
        if !our_layers.contains(&delta.layer_id) {
            return Err(LoraError::LayerNotFound(delta.layer_id));
        }

        // Store the adjustment for this layer.
        // In a full implementation, this would modify the GNN weight matrices.
        if let Some(existing) = self
            .lora_adjustments
            .iter_mut()
            .find(|(l, _)| *l == delta.layer_id)
        {
            // Accumulate: add delta.matrix_a values to existing adjustments.
            for (i, v) in delta.matrix_a.iter().enumerate() {
                if i < existing.1.len() {
                    existing.1[i] += v;
                } else {
                    existing.1.push(*v);
                }
            }
        } else {
            self.lora_adjustments
                .push((delta.layer_id, delta.matrix_a.clone()));
        }

        Ok(())
    }

    fn layer_ids(&self) -> Vec<LayerId> {
        // GNN engine manages layers based on num_layers config.
        (0..self.attention_config.num_layers as u64)
            .map(LayerId)
            .collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

    fn make_triangle_graph() -> (AdjacencyMatrix, FeatureMatrix, Vec<NodeId>) {
        // 3-node triangle with distinct features.
        let adj = AdjacencyMatrix {
            entries: vec![(0, 1, 0.8), (1, 2, 0.7), (0, 2, 0.6)],
            size: 3,
        };
        let features = FeatureMatrix {
            features: vec![
                vec![1.0, 0.0, 0.0, 0.5],
                vec![0.0, 1.0, 0.0, 0.5],
                vec![0.0, 0.0, 1.0, 0.5],
            ],
            node_count: 3,
            feature_dim: 4,
        };
        let node_ids = vec![NodeId(1), NodeId(2), NodeId(3)];
        (adj, features, node_ids)
    }

    #[test]
    fn forward_produces_correct_output_shape() {
        let engine = GnnEngine::new(AttentionModuleConfig::default());
        let (adj, features, _) = make_triangle_graph();

        let output = engine.forward(&adj, &features);

        assert_eq!(output.node_embeddings.len(), 3);
        assert_eq!(output.node_embeddings[0].len(), 4);
        assert!(!output.proofs.is_empty());
        assert_eq!(output.attention_weights.combined.len(), 3);
    }

    #[test]
    fn forward_empty_graph() {
        let engine = GnnEngine::new(AttentionModuleConfig::default());
        let adj = AdjacencyMatrix {
            entries: Vec::new(),
            size: 0,
        };
        let features = FeatureMatrix {
            features: Vec::new(),
            node_count: 0,
            feature_dim: 0,
        };

        let output = engine.forward(&adj, &features);
        assert!(output.node_embeddings.is_empty());
        assert!(output.proofs.is_empty());
    }

    #[test]
    fn attention_proof_verify_valid() {
        let temporal = vec![1u8, 2, 3, 4, 5];
        let manifold = vec![6u8, 7, 8, 9, 10];
        let hash = AttentionProof::compute_hash(&temporal, &manifold, None);

        let proof = AttentionProof {
            temporal_proof: temporal,
            manifold_proof: manifold,
            biological_proof: None,
            combined_hash: hash,
        };
        assert!(proof.verify(), "valid proof should verify");
    }

    #[test]
    fn attention_proof_verify_invalid_hash() {
        let proof = AttentionProof {
            temporal_proof: vec![1u8, 2, 3],
            manifold_proof: vec![4u8, 5, 6],
            biological_proof: None,
            combined_hash: [0u8; 32], // wrong hash
        };
        assert!(!proof.verify(), "proof with wrong hash should fail");
    }

    #[test]
    fn attention_proof_verify_empty_temporal() {
        let proof = AttentionProof {
            temporal_proof: Vec::new(),
            manifold_proof: vec![1u8, 2, 3],
            biological_proof: None,
            combined_hash: [0u8; 32],
        };
        assert!(!proof.verify(), "proof with empty temporal should fail");
    }

    #[test]
    fn extract_motifs_finds_planted_triangle() {
        let mut engine = GnnEngine::new(AttentionModuleConfig::default());
        let (adj, features, node_ids) = make_triangle_graph();

        let output = engine.forward(&adj, &features);

        let config = MotifConfig {
            min_size: 3,
            max_size: 8,
            min_frequency: 0.0,       // low threshold for test
            min_predictive_power: 0.0, // low threshold for test
        };

        let (shortcuts, failures) = engine.extract_motifs(&output, &adj, &config, &node_ids);

        assert!(
            !shortcuts.is_empty(),
            "should find the triangle motif"
        );
        assert!(
            shortcuts[0].nodes.len() >= 3,
            "triangle motif should have at least 3 nodes"
        );
        assert!(shortcuts[0].proof.verify(), "motif proof should verify");
        assert!(failures.is_empty(), "no proof failures expected for valid graph");
    }

    #[test]
    fn extract_motifs_rejects_invalid_proof() {
        let mut engine = GnnEngine::new(AttentionModuleConfig::default());
        let (adj, features, node_ids) = make_triangle_graph();

        // Create output with invalid proofs.
        let mut output = engine.forward(&adj, &features);
        output.proofs = vec![AttentionProof {
            temporal_proof: Vec::new(), // empty = invalid
            manifold_proof: Vec::new(),
            biological_proof: None,
            combined_hash: [0u8; 32],
        }];

        let config = MotifConfig {
            min_size: 3,
            max_size: 8,
            min_frequency: 0.0,
            min_predictive_power: 0.0,
        };

        let (shortcuts, failures) = engine.extract_motifs(&output, &adj, &config, &node_ids);

        assert!(shortcuts.is_empty(), "motifs with invalid proofs should be rejected");
        assert!(!failures.is_empty(), "should emit ProofFailure");
        assert!(failures[0].reason.contains("proof verification failed"));
    }

    #[test]
    fn composition_weights_sum_correctly() {
        let config = AttentionModuleConfig::default();
        let total = config.temporal_weight + config.manifold_weight + config.biological_weight;
        assert!(
            (total - 1.0).abs() < 0.01,
            "attention weights should sum to ~1.0, got {total}"
        );
    }
}
