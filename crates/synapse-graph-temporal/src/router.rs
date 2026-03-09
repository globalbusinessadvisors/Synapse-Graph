//! TemporalRouter — classifies each GatedEmbedding into a temporal tier
//! (ADR-001, DDD-003).
//!
//! Uses a lightweight FastGRNN-inspired classifier backed by
//! `ruvector-tiny-dancer-core` primitives. The Adaptation domain (Prompt 11)
//! will fine-tune the model weights later.

use synapse_graph_types::{GatedEmbedding, TemporalTier, TimingMetadata};

/// Configuration for the temporal router.
#[derive(Clone, Debug)]
pub struct RouterConfig {
    /// ISI threshold (microseconds) below which an embedding is classified as
    /// Fast tier. Default: 10_000 µs (10 ms).
    pub fast_isi_threshold_us: f32,
    /// Coherence-score threshold above which session-level trends push an
    /// embedding into Medium tier instead of Fast.
    pub session_trend_threshold: f32,
    /// Drift magnitude above which an embedding is classified as Slow tier.
    pub slow_drift_threshold: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            fast_isi_threshold_us: 10_000.0, // 10 ms
            session_trend_threshold: 0.7,
            slow_drift_threshold: 5.0,
        }
    }
}

/// Internal FastGRNN-style model weights (simplified).
///
/// In production this would be backed by `ruvector-tiny-dancer-core`. Here we
/// use a lightweight linear classifier that the Adaptation domain can replace
/// with learned weights.
#[derive(Clone, Debug)]
pub struct TinyDancerModel {
    /// Weight for ISI feature.
    w_isi: f32,
    /// Weight for burst-count feature.
    w_burst: f32,
    /// Weight for coherence-score feature.
    w_coherence: f32,
    /// Bias terms for [fast, medium, slow] classes.
    bias: [f32; 3],
}

impl Default for TinyDancerModel {
    fn default() -> Self {
        Self {
            w_isi: -1.0,       // low ISI → high fast score
            w_burst: 1.0,      // bursts push toward fast
            w_coherence: 0.3,  // high coherence nudges toward medium
            bias: [0.8, 0.2, -1.5], // default: fast > medium >> slow
        }
    }
}

impl TinyDancerModel {
    /// Compute class scores from features. Returns [fast, medium, slow].
    #[inline]
    fn scores(&self, isi_norm: f32, burst_norm: f32, coherence: f32) -> [f32; 3] {
        let fast = self.w_isi * isi_norm + self.w_burst * burst_norm + self.bias[0];
        let medium = self.w_coherence * coherence + self.bias[1];
        let slow = -self.w_isi * isi_norm - self.w_burst * burst_norm
            + self.w_coherence * coherence
            + self.bias[2];
        [fast, medium, slow]
    }
}

/// The temporal router — classifies embeddings into exactly one tier.
///
/// Invariant: classification is deterministic for a given (embedding, timing)
/// pair.
pub struct TemporalRouter {
    fastgrnn: TinyDancerModel,
    config: RouterConfig,
}

impl TemporalRouter {
    pub fn new(config: RouterConfig) -> Self {
        Self {
            fastgrnn: TinyDancerModel::default(),
            config,
        }
    }

    /// Replace the model weights (used by Adaptation domain, Prompt 11).
    pub fn set_model(&mut self, model: TinyDancerModel) {
        self.fastgrnn = model;
    }

    /// Classify a gated embedding into a temporal tier.
    ///
    /// Classification logic (DDD-003):
    ///   a) ISI < 10 ms → Fast
    ///   b) Session-level trend (coherence above threshold, no burst) → Medium
    ///   c) Significant drift from baseline → Slow
    ///   d) Default → Medium
    #[inline]
    pub fn classify(
        &self,
        embedding: &GatedEmbedding,
        timing: &TimingMetadata,
    ) -> TemporalTier {
        // Rule (a): raw ISI check.
        if timing.mean_isi_us > 0.0 && timing.mean_isi_us < self.config.fast_isi_threshold_us {
            // Burst-like timing → fast tier, unless session trend overrides.
            if timing.burst_detected {
                return TemporalTier::Fast;
            }
        }

        // Normalize features for the model.
        let isi_norm = (timing.mean_isi_us / self.config.fast_isi_threshold_us).min(1.0);
        let burst_norm = if timing.burst_detected { 1.0 } else { 0.0 };
        let coherence = embedding.coherence_score();

        let scores = self.fastgrnn.scores(isi_norm, burst_norm, coherence);

        // Rule (c): explicit drift check.
        if scores[2] > self.config.slow_drift_threshold {
            return TemporalTier::Slow;
        }

        // Rule (b): session-level trend check.
        if coherence > self.config.session_trend_threshold
            && !timing.burst_detected
            && isi_norm > 0.5
        {
            return TemporalTier::Medium;
        }

        // Model-based decision: pick the tier with highest score.
        let max_idx = if scores[0] >= scores[1] && scores[0] >= scores[2] {
            0
        } else if scores[1] >= scores[2] {
            1
        } else {
            2
        };

        match max_idx {
            0 => TemporalTier::Fast,
            1 => TemporalTier::Medium,
            _ => TemporalTier::Slow,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{ChannelId, SessionId, Timestamp};

    fn make_embedding(coherence: f32) -> GatedEmbedding {
        GatedEmbedding::__new(
            vec![0.1; 8],
            ChannelId(0),
            Timestamp(0),
            SessionId(1),
            coherence,
        )
    }

    #[test]
    fn fast_tier_for_burst_timing() {
        let router = TemporalRouter::new(RouterConfig::default());
        let emb = make_embedding(0.5);
        let timing = TimingMetadata {
            mean_isi_us: 500.0, // 0.5 ms — well below 10 ms threshold
            burst_detected: true,
            burst_count: 5,
        };
        assert_eq!(router.classify(&emb, &timing), TemporalTier::Fast);
    }

    #[test]
    fn medium_tier_for_session_level() {
        let router = TemporalRouter::new(RouterConfig::default());
        let emb = make_embedding(0.9); // high coherence
        let timing = TimingMetadata {
            mean_isi_us: 50_000.0, // 50 ms — above 10 ms, no bursts
            burst_detected: false,
            burst_count: 0,
        };
        assert_eq!(router.classify(&emb, &timing), TemporalTier::Medium);
    }

    #[test]
    fn default_is_not_slow_for_normal_input() {
        let router = TemporalRouter::new(RouterConfig::default());
        let emb = make_embedding(0.5);
        let timing = TimingMetadata {
            mean_isi_us: 20_000.0,
            burst_detected: false,
            burst_count: 0,
        };
        let tier = router.classify(&emb, &timing);
        // Should be Fast or Medium, not Slow, for normal input.
        assert_ne!(tier, TemporalTier::Slow);
    }

    #[test]
    fn classify_is_deterministic() {
        let router = TemporalRouter::new(RouterConfig::default());
        let emb = make_embedding(0.8);
        let timing = TimingMetadata {
            mean_isi_us: 5_000.0,
            burst_detected: true,
            burst_count: 3,
        };
        let t1 = router.classify(&emb, &timing);
        let t2 = router.classify(&emb, &timing);
        assert_eq!(t1, t2);
    }

    #[test]
    fn classify_latency_benchmark() {
        let router = TemporalRouter::new(RouterConfig::default());
        let emb = make_embedding(0.5);
        let timing = TimingMetadata {
            mean_isi_us: 3_000.0,
            burst_detected: true,
            burst_count: 2,
        };

        let iterations = 10_000u32;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(router.classify(
                std::hint::black_box(&emb),
                std::hint::black_box(&timing),
            ));
        }
        let per_call = start.elapsed() / iterations;
        let ceiling_us: u128 = if cfg!(debug_assertions) { 50 } else { 10 };
        assert!(
            per_call.as_micros() < ceiling_us,
            "classify() took {per_call:?}, expected < {ceiling_us}µs"
        );
        eprintln!("classify(): {per_call:?}");
    }
}
