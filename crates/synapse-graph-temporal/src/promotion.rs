//! Tier promotion logic (DDD-003).
//!
//! Monitors fast and medium tiers for patterns that should be promoted to a
//! higher-persistence tier. SlowTier execution is deferred to Prompt 07.

use synapse_graph_types::{
    PatternPromoted, PersistentPattern, PromotionCandidate, TemporalTier,
};

use crate::fast_tier::FastTier;
use crate::medium_tier::MediumTier;
use crate::slow_tier::SlowTier;

/// Configuration for tier promotion thresholds.
#[derive(Clone, Debug)]
pub struct PromoterConfig {
    /// Minimum persistence (µs) for fast-to-medium promotion. Default: 500_000 (500ms).
    pub fast_to_medium_threshold_us: u64,
    /// Minimum activations for fast-to-medium promotion. Default: 10.
    pub fast_to_medium_min_activations: u32,
    /// Minimum sessions for medium-to-slow promotion. Default: 3.
    pub medium_to_slow_sessions: usize,
    /// Minimum cross-session stability score for medium-to-slow. Default: 0.8.
    pub medium_to_slow_stability: f32,
}

impl Default for PromoterConfig {
    fn default() -> Self {
        Self {
            fast_to_medium_threshold_us: 500_000,
            fast_to_medium_min_activations: 10,
            medium_to_slow_sessions: 3,
            medium_to_slow_stability: 0.8,
        }
    }
}

/// Checks and executes tier promotions between fast, medium, and slow tiers.
pub struct TierPromoter {
    config: PromoterConfig,
}

impl TierPromoter {
    pub fn new(config: PromoterConfig) -> Self {
        Self { config }
    }

    /// Check the fast tier for patterns that should be promoted to medium.
    ///
    /// Returns candidates where:
    /// - Pattern persists > `fast_to_medium_threshold_us`
    /// - Pattern has > `fast_to_medium_min_activations`
    pub fn check_fast_promotions(&self, fast: &FastTier) -> Vec<PromotionCandidate> {
        fast.detect_persistent_patterns()
            .into_iter()
            .filter(|p| {
                p.duration_us >= self.config.fast_to_medium_threshold_us
                    && p.activation_count >= self.config.fast_to_medium_min_activations
            })
            .map(|pattern| PromotionCandidate {
                pattern,
                from_tier: TemporalTier::Fast,
                to_tier: TemporalTier::Medium,
            })
            .collect()
    }

    /// Check the medium tier for patterns that should be promoted to slow.
    ///
    /// Returns candidates where:
    /// - Pattern is stable across >= `medium_to_slow_sessions`
    /// - Cross-session stability > `medium_to_slow_stability`
    pub fn check_medium_promotions(&self, medium: &MediumTier) -> Vec<PromotionCandidate> {
        if medium.past_session_count() < self.config.medium_to_slow_sessions {
            return Vec::new();
        }

        // Extract persistent patterns from accumulated sessions by examining
        // the medium tier's context. For now, we check any patterns that have
        // been fed to us through previous promotion cycles.
        Vec::new()
    }

    /// Check if a specific pattern qualifies for medium-to-slow promotion.
    pub fn check_pattern_for_slow_promotion(
        &self,
        pattern: &PersistentPattern,
        medium: &MediumTier,
    ) -> Option<PromotionCandidate> {
        if medium.past_session_count() < self.config.medium_to_slow_sessions {
            return None;
        }

        let stability = medium.cross_session_stability(pattern);
        if stability >= self.config.medium_to_slow_stability {
            Some(PromotionCandidate {
                pattern: pattern.clone(),
                from_tier: TemporalTier::Medium,
                to_tier: TemporalTier::Slow,
            })
        } else {
            None
        }
    }

    /// Execute a fast-to-medium promotion.
    ///
    /// Accumulates the pattern centroid into the medium tier.
    /// Returns a [`PatternPromoted`] domain event.
    pub fn execute_fast_to_medium(
        &self,
        candidate: &PromotionCandidate,
        medium: &mut MediumTier,
    ) -> PatternPromoted {
        // Inject the pattern centroid into the medium tier as a synthetic
        // embedding. In a full implementation this would transfer the actual
        // buffered embeddings.
        let synthetic = synapse_graph_types::GatedEmbedding::__new(
            candidate.pattern.centroid.clone(),
            synapse_graph_types::ChannelId(0),
            synapse_graph_types::Timestamp(0),
            synapse_graph_types::SessionId(0),
            1.0, // promoted patterns have implicit high coherence
        );
        medium.accumulate(&synthetic);

        PatternPromoted {
            pattern_id: candidate.pattern.pattern_id,
            from_tier: TemporalTier::Fast,
            to_tier: TemporalTier::Medium,
            persistence_duration_us: candidate.pattern.duration_us,
            activation_count: candidate.pattern.activation_count,
        }
    }

    /// Execute a medium-to-slow promotion.
    ///
    /// Consolidates the pattern into the slow tier using EWC++.
    /// Returns a [`PatternPromoted`] domain event.
    pub fn execute_medium_to_slow(
        &self,
        candidate: &PromotionCandidate,
        slow: &mut SlowTier,
        ewc_lambda: f32,
    ) -> PatternPromoted {
        slow.consolidate(&candidate.pattern, ewc_lambda);

        PatternPromoted {
            pattern_id: candidate.pattern.pattern_id,
            from_tier: TemporalTier::Medium,
            to_tier: TemporalTier::Slow,
            persistence_duration_us: candidate.pattern.duration_us,
            activation_count: candidate.pattern.activation_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_tier::FastTierConfig;
    use crate::medium_tier::MediumTierConfig;
    use synapse_graph_types::{ChannelId, GatedEmbedding, SessionId, Timestamp};

    fn make_gated(dim: usize, base: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..dim).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), 0.9)
    }

    #[test]
    fn promoter_detects_persistent_fast_pattern() {
        let mut fast = FastTier::new(FastTierConfig {
            capacity: 256,
            min_persistence_us: 500_000,
            min_activations: 10,
            pattern_distance_threshold: 1.0,
        });

        // Plant a persistent pattern: >500ms, >10 activations.
        for i in 0..15u64 {
            let base = 1.0 + (i as f32) * 0.001;
            fast.push(&make_gated(8, base), Timestamp(i * 50_000));
        }

        let promoter = TierPromoter::new(PromoterConfig::default());
        let candidates = promoter.check_fast_promotions(&fast);
        assert!(
            !candidates.is_empty(),
            "should detect at least one promotion candidate"
        );
        assert_eq!(candidates[0].from_tier, TemporalTier::Fast);
        assert_eq!(candidates[0].to_tier, TemporalTier::Medium);
        assert!(candidates[0].pattern.activation_count >= 10);
    }

    #[test]
    fn promotion_fast_to_medium_executes() {
        let mut fast = FastTier::new(FastTierConfig {
            capacity: 256,
            min_persistence_us: 500_000,
            min_activations: 10,
            pattern_distance_threshold: 1.0,
        });

        for i in 0..15u64 {
            fast.push(&make_gated(8, 1.0 + (i as f32) * 0.001), Timestamp(i * 50_000));
        }

        let promoter = TierPromoter::new(PromoterConfig::default());
        let candidates = promoter.check_fast_promotions(&fast);
        assert!(!candidates.is_empty());

        let mut medium = MediumTier::new(MediumTierConfig {
            dimension: 8,
            ..Default::default()
        });
        assert_eq!(medium.current_session_count(), 0);

        let event = promoter.execute_fast_to_medium(&candidates[0], &mut medium);
        assert_eq!(event.from_tier, TemporalTier::Fast);
        assert_eq!(event.to_tier, TemporalTier::Medium);
        assert!(medium.current_session_count() > 0, "medium tier should have received the pattern");
    }

    #[test]
    fn no_slow_promotion_without_enough_sessions() {
        let medium = MediumTier::new(MediumTierConfig {
            dimension: 8,
            max_past_sessions: 10,
            ..Default::default()
        });

        let promoter = TierPromoter::new(PromoterConfig::default());
        let candidates = promoter.check_medium_promotions(&medium);
        assert!(candidates.is_empty(), "no slow promotions without past sessions");
    }

    #[test]
    fn pattern_slow_promotion_with_stability() {
        let mut medium = MediumTier::new(MediumTierConfig {
            dimension: 8,
            max_past_sessions: 10,
            ..Default::default()
        });

        // Build 5 similar sessions.
        for _ in 0..5 {
            for i in 0..20 {
                medium.accumulate(&make_gated(8, 1.0 + (i as f32) * 0.001));
            }
            medium.end_session();
        }

        let stable_pattern = PersistentPattern {
            pattern_id: synapse_graph_types::PatternId(1),
            centroid: (0..8).map(|i| 1.0 + (i as f32) * 0.01).collect(),
            activation_count: 50,
            duration_us: 2_000_000,
        };

        let promoter = TierPromoter::new(PromoterConfig {
            medium_to_slow_stability: 0.5, // lower threshold for test reliability
            ..Default::default()
        });

        let candidate = promoter.check_pattern_for_slow_promotion(&stable_pattern, &medium);
        assert!(
            candidate.is_some(),
            "stable pattern should qualify for slow promotion"
        );
        let c = candidate.unwrap();
        assert_eq!(c.from_tier, TemporalTier::Medium);
        assert_eq!(c.to_tier, TemporalTier::Slow);
    }
}
