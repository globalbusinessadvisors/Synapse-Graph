//! AttentionScorer — min-cut gated attention scoring (DDD-005).
//!
//! Uses min-cut gated attention (not softmax) to score intents based on
//! partial pattern queries and matched shortcuts. Conceptually backed by
//! ruvector-attention and ruvector-attn-mincut.

use synapse_graph_types::{IntentId, LayerId, LoraDelta, LoraError, LoraTarget, MatchedShortcut, ScoredIntent};

/// Configuration for the AttentionScorer.
#[derive(Clone, Debug)]
pub struct ScorerConfig {
    /// Temperature for attention scaling. Default: 1.0.
    pub temperature: f32,
    /// Minimum gate threshold for min-cut gating. Default: 0.1.
    pub min_gate_threshold: f32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_gate_threshold: 0.1,
        }
    }
}

/// Scores intents using min-cut gated attention.
pub struct AttentionScorer {
    config: ScorerConfig,
    /// Accumulated LoRA weight adjustments per layer.
    lora_adjustments: Vec<(LayerId, Vec<f32>)>,
}

impl AttentionScorer {
    pub fn new(config: ScorerConfig) -> Self {
        Self {
            config,
            lora_adjustments: Vec::new(),
        }
    }

    /// Access LoRA adjustments (for testing/inspection).
    pub fn lora_adjustments(&self) -> &[(LayerId, Vec<f32>)] {
        &self.lora_adjustments
    }

    /// Score intents based on a query pattern and matched shortcuts.
    ///
    /// Uses min-cut gated attention per DDD-005:
    /// - query = partial_pattern vector
    /// - keys = shortcut node centroids (averaged)
    /// - values = intent associations
    /// - gate = dynamic min-cut (thresholds low-confidence matches)
    pub fn score(
        &self,
        query: &[f32],
        matched: &[MatchedShortcut],
    ) -> Vec<ScoredIntent> {
        let mut scored = Vec::new();

        for m in matched {
            // Compute attention weight: dot-product attention with temperature.
            let key = shortcut_centroid(&m.shortcut);
            let raw_attn = dot_product(query, &key) / self.config.temperature;

            // Min-cut gating: zero out low-attention shortcuts.
            let gate = if raw_attn >= self.config.min_gate_threshold {
                raw_attn
            } else {
                0.0
            };

            if gate <= 0.0 {
                continue;
            }

            // Map each intent association to a ScoredIntent.
            if m.shortcut.intent_associations.is_empty() {
                // No explicit associations — use the shortcut's predictive power
                // as a generic intent score with a synthetic intent ID.
                scored.push(ScoredIntent {
                    intent_id: IntentId(m.shortcut.id.0),
                    confidence: gate * m.shortcut.predictive_power,
                    shortcut_id: m.shortcut.id,
                    match_distance: m.distance,
                });
            } else {
                for assoc in &m.shortcut.intent_associations {
                    scored.push(ScoredIntent {
                        intent_id: assoc.intent_id,
                        confidence: gate * assoc.strength,
                        shortcut_id: m.shortcut.id,
                        match_distance: m.distance,
                    });
                }
            }
        }

        // Normalize confidences.
        let total: f32 = scored.iter().map(|s| s.confidence).sum();
        if total > 0.0 {
            for s in &mut scored {
                s.confidence /= total;
            }
        }

        scored
    }

    /// Return attention weights for the matched shortcuts (for diagnostics).
    pub fn compute_attention_weights(
        &self,
        query: &[f32],
        matched: &[MatchedShortcut],
    ) -> Vec<f32> {
        matched
            .iter()
            .map(|m| {
                let key = shortcut_centroid(&m.shortcut);
                let raw = dot_product(query, &key) / self.config.temperature;
                if raw >= self.config.min_gate_threshold {
                    raw
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Compute the centroid of a shortcut's node set (average of node IDs mapped
/// to feature vectors). For simplicity, uses the first node's features from
/// the shortcut's edges.
fn shortcut_centroid(shortcut: &synapse_graph_types::CognitiveShortcut) -> Vec<f32> {
    // Use the shortcut's edges to approximate a centroid.
    // In a real implementation we'd look up actual node features.
    // Here we use a stable hash-based proxy per node ID.
    let dim = 8; // default feature dimension
    let n = shortcut.nodes.len().max(1) as f32;
    let mut centroid = vec![0.0f32; dim];
    for node_id in &shortcut.nodes {
        let seed = node_id.0 as f32;
        for d in 0..dim {
            centroid[d] += (seed * 0.1 + d as f32 * 0.3).sin() / n;
        }
    }
    centroid
}

impl LoraTarget for AttentionScorer {
    fn apply_lora_delta(&mut self, delta: &LoraDelta) -> Result<(), LoraError> {
        let our_layers = self.layer_ids();
        if !our_layers.contains(&delta.layer_id) {
            return Err(LoraError::LayerNotFound(delta.layer_id));
        }

        // Store the adjustment for this layer.
        if let Some(existing) = self
            .lora_adjustments
            .iter_mut()
            .find(|(l, _)| *l == delta.layer_id)
        {
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
        // AttentionScorer manages attention head layers.
        vec![LayerId(0), LayerId(1)]
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .max(0.0) // clamp to non-negative for attention
}
