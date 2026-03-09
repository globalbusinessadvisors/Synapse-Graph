//! TemporalReweighter — adjusts intent scores based on temporal context (DDD-005).
//!
//! Boosts intents consistent with the current temporal context and penalizes
//! contradictory ones.

use synapse_graph_types::{ScoredIntent, TemporalContext};

/// Configuration for the TemporalReweighter.
#[derive(Clone, Debug)]
pub struct ReweightConfig {
    /// Multiplicative boost factor for temporal consistency. Default: 0.5.
    ///
    /// confidence *= (1.0 + boost_factor * temporal_consistency)
    pub boost_factor: f32,
}

impl Default for ReweightConfig {
    fn default() -> Self {
        Self { boost_factor: 0.5 }
    }
}

/// Reweights scored intents based on temporal context.
pub struct TemporalReweighter {
    config: ReweightConfig,
}

impl TemporalReweighter {
    pub fn new(config: ReweightConfig) -> Self {
        Self { config }
    }

    /// Reweight scored intents using temporal context blending.
    ///
    /// For each ScoredIntent:
    ///   temporal_consistency = cosine_similarity(intent_pattern, context.blended())
    ///   confidence *= (1.0 + boost_factor * temporal_consistency)
    ///
    /// Returns the total temporal influence (average consistency).
    pub fn reweight(
        &self,
        scored: &mut [ScoredIntent],
        context: &TemporalContext,
    ) -> f32 {
        let blended = context.blended();
        if blended.is_empty() || scored.is_empty() {
            return 0.0;
        }

        let mut total_consistency = 0.0f32;

        for intent in scored.iter_mut() {
            // Generate an intent pattern proxy from the intent_id.
            // In a full implementation this would be a learned embedding per intent.
            let intent_pattern = intent_pattern_proxy(intent.intent_id.0, blended.len());

            let consistency = cosine_similarity(&intent_pattern, &blended);
            total_consistency += consistency;

            // Apply boost/penalty.
            intent.confidence *= 1.0 + self.config.boost_factor * consistency;
        }

        // Re-normalize confidences.
        let total: f32 = scored.iter().map(|s| s.confidence).sum();
        if total > 0.0 {
            for s in scored.iter_mut() {
                s.confidence /= total;
            }
        }

        total_consistency / scored.len() as f32
    }
}

/// Generate a deterministic intent pattern proxy from an intent ID.
fn intent_pattern_proxy(intent_id: u64, dim: usize) -> Vec<f32> {
    let seed = intent_id as f32;
    (0..dim)
        .map(|d| (seed * 0.7 + d as f32 * 0.3).sin())
        .collect()
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
