//! TemporalLearner — aggregate root for the temporal learning domain (DDD-003).
//!
//! Wires together the TemporalRouter, FastTier, MediumTier, SlowTier, and
//! TierPromoter into a single entry point for the application layer.

use synapse_graph_types::{
    EmbeddingCluster, GatedEmbedding, MaintenanceResult, PatternId, PersistentPattern,
    TemporalContext, TemporalSequence, TemporalTier, TierWeights, TimingMetadata,
};

use crate::fast_tier::{FastTier, FastTierConfig};
use crate::medium_tier::{MediumTier, MediumTierConfig};
use crate::promotion::{PromoterConfig, TierPromoter};
use crate::router::{RouterConfig, TemporalRouter};
use crate::slow_tier::{SlowTier, SlowTierConfig};

/// Configuration for the TemporalLearner aggregate.
#[derive(Clone, Debug)]
pub struct TemporalConfig {
    pub router: RouterConfig,
    pub fast: FastTierConfig,
    pub medium: MediumTierConfig,
    pub slow: SlowTierConfig,
    pub promoter: PromoterConfig,
    /// EWC lambda for slow-tier consolidation. Default: 0.4.
    pub ewc_lambda: f32,
    /// Default weights for context blending.
    pub default_weights: TierWeights,
    /// Context window (µs) for fast-tier blending. Default: 100_000 (100ms).
    pub fast_context_window_us: u64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            router: RouterConfig::default(),
            fast: FastTierConfig::default(),
            medium: MediumTierConfig::default(),
            slow: SlowTierConfig::default(),
            promoter: PromoterConfig::default(),
            ewc_lambda: 0.4,
            default_weights: TierWeights::default(),
            fast_context_window_us: 100_000,
        }
    }
}

/// The TemporalLearner aggregate root — unified entry point for temporal learning.
pub struct TemporalLearner {
    router: TemporalRouter,
    fast: FastTier,
    medium: MediumTier,
    slow: SlowTier,
    promoter: TierPromoter,
    config: TemporalConfig,
}

impl TemporalLearner {
    pub fn new(config: TemporalConfig) -> Self {
        let router = TemporalRouter::new(config.router.clone());
        let fast = FastTier::new(config.fast.clone());
        let medium = MediumTier::new(config.medium.clone());
        let slow = SlowTier::new(config.slow.clone());
        let promoter = TierPromoter::new(config.promoter.clone());
        Self {
            router,
            fast,
            medium,
            slow,
            promoter,
            config,
        }
    }

    /// Accumulate a gated embedding into the appropriate temporal tier.
    ///
    /// Steps:
    /// 1. Router classifies the embedding into a tier.
    /// 2. The embedding is dispatched to the corresponding tier.
    ///
    /// Returns which tier the embedding was routed to.
    pub fn accumulate(
        &mut self,
        embedding: &GatedEmbedding,
        timing: &TimingMetadata,
    ) -> TemporalTier {
        let tier = self.router.classify(embedding, timing);

        match tier {
            TemporalTier::Fast => {
                self.fast.push(embedding, embedding.timestamp());
            }
            TemporalTier::Medium => {
                self.medium.accumulate(embedding);
            }
            TemporalTier::Slow => {
                // For direct slow-tier routing, create a synthetic pattern
                // from the embedding and consolidate.
                let pattern = PersistentPattern {
                    pattern_id: PatternId(0),
                    centroid: embedding.vector().to_vec(),
                    activation_count: 1,
                    duration_us: 0,
                };
                self.slow.consolidate(&pattern, self.config.ewc_lambda);
            }
        }

        tier
    }

    /// Get the blended temporal context across all three tiers.
    pub fn get_context(&self, weights: TierWeights) -> TemporalContext {
        // Fast component: mean of recent embeddings.
        let fast_embeddings = self.fast.recent_context(self.config.fast_context_window_us);
        let fast_component = if fast_embeddings.is_empty() {
            vec![0.0; self.config.medium.dimension]
        } else {
            let dim = fast_embeddings[0].vector().len();
            let mut mean = vec![0.0f32; dim];
            for emb in &fast_embeddings {
                for (i, &v) in emb.vector().iter().enumerate() {
                    if i < mean.len() {
                        mean[i] += v;
                    }
                }
            }
            let n = fast_embeddings.len() as f32;
            for v in &mut mean {
                *v /= n;
            }
            mean
        };

        let medium_component = self.medium.session_context();
        let slow_component = self.slow.baseline_context();

        TemporalContext {
            fast_component,
            medium_component,
            slow_component,
            weights,
        }
    }

    /// Extract temporal sequences from the medium tier.
    pub fn extract_sequences(&self, cluster: &EmbeddingCluster) -> Vec<TemporalSequence> {
        self.medium.extract_sequences(cluster)
    }

    /// Run maintenance: check promotions, execute them, and manage resources.
    ///
    /// Returns a [`MaintenanceResult`] summarizing what happened.
    pub fn maintain(&mut self) -> MaintenanceResult {
        let mut promoted = 0usize;
        let evicted = 0usize;

        // (a) Check and execute fast-to-medium promotions.
        let fast_candidates = self.promoter.check_fast_promotions(&self.fast);
        for candidate in &fast_candidates {
            self.promoter
                .execute_fast_to_medium(candidate, &mut self.medium);
            promoted += 1;
        }

        // (b) Check and execute medium-to-slow promotions.
        // For each fast-promoted pattern that now exists in medium, check if
        // it qualifies for slow promotion.
        for candidate in &fast_candidates {
            if let Some(slow_candidate) = self
                .promoter
                .check_pattern_for_slow_promotion(&candidate.pattern, &self.medium)
            {
                self.execute_medium_to_slow(&slow_candidate.pattern);
                promoted += 1;
            }
        }

        // Also check medium promotions directly.
        let medium_candidates = self.promoter.check_medium_promotions(&self.medium);
        for candidate in &medium_candidates {
            self.execute_medium_to_slow(&candidate.pattern);
            promoted += 1;
        }

        MaintenanceResult { promoted, evicted }
    }

    /// Execute a medium-to-slow promotion by consolidating into the slow tier.
    fn execute_medium_to_slow(&mut self, pattern: &PersistentPattern) {
        self.slow.consolidate(pattern, self.config.ewc_lambda);
    }

    /// End the current session: archive the medium tier's session tensor.
    pub fn end_session(&mut self) {
        self.medium.end_session();
    }

    /// Access the temporal router.
    pub fn router(&self) -> &TemporalRouter {
        &self.router
    }

    /// Access the fast tier (for testing/inspection).
    pub fn fast(&self) -> &FastTier {
        &self.fast
    }

    /// Access the medium tier (for testing/inspection).
    pub fn medium(&self) -> &MediumTier {
        &self.medium
    }

    /// Access the slow tier (for testing/inspection).
    pub fn slow(&self) -> &SlowTier {
        &self.slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{ChannelId, DriftDetected, SessionId, Timestamp};

    fn make_gated(dim: usize, base: f32, coherence: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..dim).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), coherence)
    }

    fn make_burst_timing() -> TimingMetadata {
        TimingMetadata {
            mean_isi_us: 500.0, // 0.5ms — well below 10ms
            burst_detected: true,
            burst_count: 5,
        }
    }

    fn make_medium_timing() -> TimingMetadata {
        TimingMetadata {
            mean_isi_us: 50_000.0, // 50ms
            burst_detected: false,
            burst_count: 0,
        }
    }

    #[test]
    fn accumulate_routes_burst_to_fast() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                ..Default::default()
            },
            ..Default::default()
        });

        let emb = make_gated(8, 1.0, 0.5);
        let tier = learner.accumulate(&emb, &make_burst_timing());
        assert_eq!(tier, TemporalTier::Fast);
        assert_eq!(learner.fast().len(), 1);
    }

    #[test]
    fn accumulate_routes_session_to_medium() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                ..Default::default()
            },
            ..Default::default()
        });

        let emb = make_gated(8, 1.0, 0.9); // high coherence
        let tier = learner.accumulate(&emb, &make_medium_timing());
        assert_eq!(tier, TemporalTier::Medium);
        assert!(learner.medium().current_session_count() > 0);
    }

    #[test]
    fn get_context_blends_three_tiers() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            fast: FastTierConfig {
                capacity: 256,
                ..Default::default()
            },
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                ..Default::default()
            },
            fast_context_window_us: u64::MAX,
            ..Default::default()
        });

        // Add data to fast tier.
        let fast_emb = make_gated(8, 1.0, 0.5);
        learner.accumulate(&fast_emb, &make_burst_timing());

        // Add data to medium tier.
        let med_emb = make_gated(8, 2.0, 0.9);
        learner.accumulate(&med_emb, &make_medium_timing());

        // Consolidate into slow tier.
        let pattern = PersistentPattern {
            pattern_id: PatternId(1),
            centroid: (0..8).map(|i| 3.0 + (i as f32) * 0.01).collect(),
            activation_count: 20,
            duration_us: 1_000_000,
        };
        learner.slow.consolidate(&pattern, 0.4);

        let weights = TierWeights {
            fast: 0.5,
            medium: 0.3,
            slow: 0.2,
        };
        let ctx = learner.get_context(weights);

        assert_eq!(ctx.weights, weights);
        assert!(!ctx.fast_component.is_empty());
        assert!(!ctx.medium_component.is_empty());
        assert!(!ctx.slow_component.is_empty());

        let blended = ctx.blended();
        assert_eq!(blended.len(), 8);
        // Blended should be non-zero since all tiers have data.
        assert!(blended.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn maintain_promotes_persistent_fast_pattern() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            fast: FastTierConfig {
                capacity: 256,
                min_persistence_us: 500_000,
                min_activations: 10,
                pattern_distance_threshold: 1.0,
            },
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                ..Default::default()
            },
            ..Default::default()
        });

        // Plant a persistent pattern in fast tier.
        for i in 0..15u64 {
            let emb = GatedEmbedding::__new(
                (0..8).map(|j| 1.0 + (j as f32) * 0.01 + (i as f32) * 0.001).collect(),
                ChannelId(0),
                Timestamp(i * 50_000),
                SessionId(1),
                0.5,
            );
            learner.fast.push(&emb, Timestamp(i * 50_000));
        }

        let result = learner.maintain();
        assert!(
            result.promoted > 0,
            "should promote at least one persistent pattern"
        );
        assert!(
            learner.medium().current_session_count() > 0,
            "medium tier should receive promoted pattern"
        );
    }

    #[test]
    fn maintain_promotes_stable_medium_to_slow() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            fast: FastTierConfig {
                capacity: 256,
                min_persistence_us: 500_000,
                min_activations: 10,
                pattern_distance_threshold: 1.0,
            },
            medium: MediumTierConfig {
                dimension: 8,
                max_past_sessions: 10,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                ..Default::default()
            },
            promoter: PromoterConfig {
                medium_to_slow_sessions: 3,
                medium_to_slow_stability: 0.5,
                ..Default::default()
            },
            ..Default::default()
        });

        // Build 5 similar sessions in medium tier.
        for _ in 0..5 {
            for i in 0..20 {
                let emb = make_gated(8, 1.0 + (i as f32) * 0.001, 0.9);
                learner.medium.accumulate(&emb);
            }
            learner.medium.end_session();
        }

        // Also plant a persistent fast-tier pattern that matches the medium sessions.
        for i in 0..15u64 {
            let emb = GatedEmbedding::__new(
                (0..8).map(|j| 1.0 + (j as f32) * 0.01 + (i as f32) * 0.001).collect(),
                ChannelId(0),
                Timestamp(i * 50_000),
                SessionId(1),
                0.5,
            );
            learner.fast.push(&emb, Timestamp(i * 50_000));
        }

        let result = learner.maintain();
        // Fast-to-medium promotion should happen, and the pattern should also be
        // checked for medium-to-slow promotion.
        assert!(result.promoted > 0);
    }

    #[test]
    fn drift_detection_fires_for_divergent_pattern() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                drift_threshold: 3.0,
                ..Default::default()
            },
            ..Default::default()
        });

        // Consolidate a baseline pattern.
        let pattern = PersistentPattern {
            pattern_id: PatternId(1),
            centroid: vec![1.0; 8],
            activation_count: 20,
            duration_us: 1_000_000,
        };
        learner.slow.consolidate(&pattern, 0.4);

        // Check drift with a distant embedding.
        let far = vec![100.0; 8];
        let drift = learner.slow().detect_drift(&far);
        assert!(drift.is_some(), "should detect drift for distant embedding");
    }

    #[test]
    fn requantize_reduces_slow_tier_bits() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                quantization_bits: 4,
                ..Default::default()
            },
            ..Default::default()
        });

        // Record a drift for requantization to affect.
        let drift = DriftDetected {
            drift_magnitude: 5.0,
            affected_channels: vec![],
            drift_direction: vec![0.1, -0.3, 0.5, -0.7, 0.2, -0.4, 0.6, -0.8],
        };
        learner.slow.record_drift(&drift, Timestamp(1000));

        assert_eq!(learner.slow().quantization_bits(), 4);
        learner.slow.requantize(2);
        assert_eq!(learner.slow().quantization_bits(), 2);
    }

    #[test]
    fn end_session_archives_medium() {
        let mut learner = TemporalLearner::new(TemporalConfig {
            medium: MediumTierConfig {
                dimension: 8,
                ..Default::default()
            },
            slow: SlowTierConfig {
                dimension: 8,
                ..Default::default()
            },
            ..Default::default()
        });

        // Accumulate some data into medium tier.
        for i in 0..10 {
            let emb = make_gated(8, 1.0 + (i as f32) * 0.01, 0.9);
            learner.accumulate(&emb, &make_medium_timing());
        }

        assert!(learner.medium().current_session_count() > 0);
        assert_eq!(learner.medium().past_session_count(), 0);

        learner.end_session();

        assert_eq!(learner.medium().current_session_count(), 0);
        assert_eq!(learner.medium().past_session_count(), 1);
    }

    #[test]
    fn context_blending_weights_are_applied() {
        let ctx = TemporalContext {
            fast_component: vec![1.0, 2.0, 3.0],
            medium_component: vec![4.0, 5.0, 6.0],
            slow_component: vec![7.0, 8.0, 9.0],
            weights: TierWeights {
                fast: 0.5,
                medium: 0.3,
                slow: 0.2,
            },
        };

        let blended = ctx.blended();
        // blended[0] = 1.0*0.5 + 4.0*0.3 + 7.0*0.2 = 0.5 + 1.2 + 1.4 = 3.1
        assert!((blended[0] - 3.1).abs() < 0.001);
        // blended[1] = 2.0*0.5 + 5.0*0.3 + 8.0*0.2 = 1.0 + 1.5 + 1.6 = 4.1
        assert!((blended[1] - 4.1).abs() < 0.001);
        // blended[2] = 3.0*0.5 + 6.0*0.3 + 9.0*0.2 = 1.5 + 1.8 + 1.8 = 5.1
        assert!((blended[2] - 5.1).abs() < 0.001);
    }
}
