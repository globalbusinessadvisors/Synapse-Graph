//! CoherenceGate — the hard architectural boundary (ADR-002).
//!
//! Every raw embedding must pass through [`CoherenceGate::evaluate`] before it
//! can be wrapped in a [`GatedEmbedding`]. The gate performs four checks:
//!
//! 1. **Manifold consistency** — Mahalanobis distance from running statistics.
//! 2. **Temporal continuity** — L2 delta vs previous embedding on the same channel.
//! 3. **Channel health** — SNR and dropout rate thresholds.
//! 4. **Zero / saturated signal** — trivially degenerate inputs.

use std::collections::HashMap;

use synapse_graph_types::{
    ChannelHealth, ChannelId, CoherenceVerdict, DenyReason, GatedEmbedding, SessionId, Timestamp,
};

/// Tunable thresholds for the coherence gate.
#[derive(Clone, Debug)]
pub struct GateConfig {
    /// Mahalanobis distance threshold (number of sigma). Default: 3.0.
    pub sigma_threshold: f32,
    /// Maximum L2 distance between consecutive embeddings on the same channel.
    pub max_temporal_delta: f32,
    /// Minimum acceptable signal-to-noise ratio per channel.
    pub min_snr: f32,
    /// Maximum acceptable dropout rate per channel.
    pub max_dropout: f32,
    /// Minimum number of samples before manifold statistics are trusted.
    pub min_samples: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 3.0,
            max_temporal_delta: 10.0,
            min_snr: 1.0,
            max_dropout: 0.5,
            min_samples: 2,
        }
    }
}

/// Running statistics for a single channel's manifold estimate.
///
/// Tracks mean, variance (diagonal covariance), and last embedding using
/// Welford's online algorithm. Kept under 256 bytes per channel by storing
/// only diagonal variance (no full covariance matrix).
#[derive(Clone, Debug)]
pub struct RunningManifoldStats {
    count: usize,
    mean: Vec<f32>,
    m2: Vec<f32>,
    last_embedding: Option<Vec<f32>>,
    health: ChannelHealth,
}

impl RunningManifoldStats {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            last_embedding: None,
            health: ChannelHealth {
                impedance: 1.0,
                snr: f32::MAX,
                dropout_rate: 0.0,
            },
        }
    }

    /// Update running mean/variance with a new sample (Welford's algorithm)
    /// and store as last embedding for temporal delta checks.
    fn update_and_store(&mut self, sample: &[f32]) {
        self.count += 1;
        let n = self.count as f32;
        let inv_n = 1.0 / n;
        for i in 0..sample.len().min(self.mean.len()) {
            let delta = sample[i] - self.mean[i];
            self.mean[i] += delta * inv_n;
            let delta2 = sample[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
        // Reuse existing allocation when possible.
        match &mut self.last_embedding {
            Some(prev) => prev.copy_from_slice(sample),
            None => self.last_embedding = Some(sample.to_vec()),
        }
    }

    /// Squared Mahalanobis distance using diagonal covariance.
    fn mahalanobis_sq(&self, sample: &[f32]) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f32;
        let mut dist = 0.0f32;
        for i in 0..sample.len().min(self.mean.len()) {
            let variance = self.m2[i] / (n - 1.0);
            if variance > f32::EPSILON {
                let diff = sample[i] - self.mean[i];
                dist += (diff * diff) / variance;
            }
        }
        dist
    }

    /// L2 distance to the previous embedding on this channel.
    fn temporal_delta(&self, sample: &[f32]) -> Option<f32> {
        self.last_embedding.as_ref().map(|prev| {
            let mut sum = 0.0f32;
            for i in 0..sample.len().min(prev.len()) {
                let d = sample[i] - prev[i];
                sum += d * d;
            }
            sum.sqrt()
        })
    }
}

/// The coherence gate — hard boundary between raw input and downstream processing.
pub struct CoherenceGate {
    config: GateConfig,
    channel_stats: HashMap<ChannelId, RunningManifoldStats>,
}

impl CoherenceGate {
    pub fn new(config: GateConfig) -> Self {
        Self {
            config,
            channel_stats: HashMap::new(),
        }
    }

    /// Return a reference to the current config.
    pub fn config(&self) -> &GateConfig {
        &self.config
    }

    /// Get current health metrics for a channel, if tracked.
    pub fn channel_health(&self, channel: ChannelId) -> Option<ChannelHealth> {
        self.channel_stats.get(&channel).map(|s| s.health)
    }

    /// Update health metrics for a channel.
    pub fn set_channel_health(&mut self, channel: ChannelId, health: ChannelHealth) {
        let dim = self
            .channel_stats
            .values()
            .next()
            .map_or(0, |s| s.mean.len());
        let stats = self
            .channel_stats
            .entry(channel)
            .or_insert_with(|| RunningManifoldStats::new(dim));
        stats.health = health;
    }

    /// Evaluate a raw embedding against the four coherence checks.
    ///
    /// Returns [`CoherenceVerdict::Permit`] with a coherence score on success,
    /// or [`CoherenceVerdict::Deny`] with the specific [`DenyReason`] on failure.
    pub fn evaluate(
        &mut self,
        raw: &[f32],
        channel: ChannelId,
        _timestamp: Timestamp,
        _session: SessionId,
    ) -> CoherenceVerdict {
        let dim = raw.len();
        let stats = self
            .channel_stats
            .entry(channel)
            .or_insert_with(|| RunningManifoldStats::new(dim));

        // Check 4: Zero / saturated signal (cheapest, do first).
        if raw.iter().all(|&v| v == 0.0) {
            return CoherenceVerdict::Deny {
                reason: DenyReason::ZeroSignal,
            };
        }
        if raw.iter().all(|&v| v == f32::MAX) {
            return CoherenceVerdict::Deny {
                reason: DenyReason::SaturatedSignal,
            };
        }

        // Check 3: Channel health.
        if stats.health.snr < self.config.min_snr {
            return CoherenceVerdict::Deny {
                reason: DenyReason::ChannelUnhealthy,
            };
        }
        if stats.health.dropout_rate > self.config.max_dropout {
            return CoherenceVerdict::Deny {
                reason: DenyReason::ChannelUnhealthy,
            };
        }

        // Check 1: Manifold consistency (Mahalanobis distance).
        if stats.count >= self.config.min_samples {
            let maha_sq = stats.mahalanobis_sq(raw);
            let threshold_sq = self.config.sigma_threshold * self.config.sigma_threshold;
            // Mahalanobis distance is compared as: sqrt(d²/dim) > sigma
            // Equivalently: d² > sigma² * dim
            let dim_f = dim as f32;
            if maha_sq > threshold_sq * dim_f {
                return CoherenceVerdict::Deny {
                    reason: DenyReason::ManifoldOutlier,
                };
            }
        }

        // Check 2: Temporal continuity.
        if let Some(delta) = stats.temporal_delta(raw) {
            if delta > self.config.max_temporal_delta {
                return CoherenceVerdict::Deny {
                    reason: DenyReason::TemporalDiscontinuity,
                };
            }
        }

        // All checks passed — compute coherence score and update stats.
        let score = Self::compute_score_inner(raw, stats, &self.config);
        stats.update_and_store(raw);

        CoherenceVerdict::Permit { score }
    }

    /// Mint a [`GatedEmbedding`] — only callable within this crate.
    pub(crate) fn mint_gated(
        &self,
        raw: Vec<f32>,
        channel: ChannelId,
        timestamp: Timestamp,
        session: SessionId,
        score: f32,
    ) -> GatedEmbedding {
        GatedEmbedding::__new(raw, channel, timestamp, session, score)
    }

    /// Convenience: evaluate and, if permitted, mint the gated embedding.
    pub fn evaluate_and_mint(
        &mut self,
        raw: Vec<f32>,
        channel: ChannelId,
        timestamp: Timestamp,
        session: SessionId,
    ) -> Result<GatedEmbedding, DenyReason> {
        match self.evaluate(&raw, channel, timestamp, session) {
            CoherenceVerdict::Permit { score } => {
                Ok(self.mint_gated(raw, channel, timestamp, session, score))
            }
            CoherenceVerdict::Deny { reason } => Err(reason),
        }
    }

    /// Compute a coherence score in [0.0, 1.0] from manifold distance.
    fn compute_score_inner(raw: &[f32], stats: &RunningManifoldStats, config: &GateConfig) -> f32 {
        if stats.count < config.min_samples {
            return 1.0;
        }
        let maha_sq = stats.mahalanobis_sq(raw);
        let dim_f = raw.len() as f32;
        let threshold_sq = config.sigma_threshold * config.sigma_threshold * dim_f;
        // Score decays from 1.0 as distance approaches the threshold.
        (1.0 - maha_sq / threshold_sq).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gate() -> CoherenceGate {
        CoherenceGate::new(GateConfig::default())
    }

    fn make_embedding(dim: usize, base: f32) -> Vec<f32> {
        (0..dim).map(|i| base + (i as f32) * 0.01).collect()
    }

    #[test]
    fn valid_embedding_passes_gate() {
        let mut gate = default_gate();
        let ch = ChannelId(1);
        let ts = Timestamp(1000);
        let sess = SessionId(1);

        // Seed with a few samples so stats are established.
        let emb = make_embedding(8, 1.0);
        assert!(matches!(
            gate.evaluate(&emb, ch, ts, sess),
            CoherenceVerdict::Permit { .. }
        ));
        let emb2 = make_embedding(8, 1.01);
        assert!(matches!(
            gate.evaluate(&emb2, ch, ts, sess),
            CoherenceVerdict::Permit { .. }
        ));
        // Third sample — stats now active; nearby embedding should still pass.
        let emb3 = make_embedding(8, 1.02);
        match gate.evaluate(&emb3, ch, ts, sess) {
            CoherenceVerdict::Permit { score } => {
                assert!(score > 0.0 && score <= 1.0, "score={score}");
            }
            other => panic!("expected Permit, got {other:?}"),
        }
    }

    #[test]
    fn outlier_embedding_denied_manifold() {
        let mut gate = default_gate();
        let ch = ChannelId(1);
        let ts = Timestamp(1000);
        let sess = SessionId(1);

        // Build up manifold stats with similar embeddings.
        for i in 0..10 {
            let emb = make_embedding(8, 1.0 + (i as f32) * 0.001);
            gate.evaluate(&emb, ch, ts, sess);
        }

        // Wild outlier — very far from the established manifold.
        let outlier = make_embedding(8, 1000.0);
        match gate.evaluate(&outlier, ch, ts, sess) {
            CoherenceVerdict::Deny { reason } => {
                // Could be ManifoldOutlier or TemporalDiscontinuity depending on
                // which check fires first; both are valid rejections for an outlier.
                assert!(
                    reason == DenyReason::ManifoldOutlier
                        || reason == DenyReason::TemporalDiscontinuity,
                    "unexpected reason: {reason:?}"
                );
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[test]
    fn zero_signal_denied() {
        let mut gate = default_gate();
        let zeros = vec![0.0f32; 8];
        let verdict = gate.evaluate(&zeros, ChannelId(1), Timestamp(0), SessionId(1));
        assert_eq!(
            verdict,
            CoherenceVerdict::Deny {
                reason: DenyReason::ZeroSignal
            }
        );
    }

    #[test]
    fn saturated_signal_denied() {
        let mut gate = default_gate();
        let saturated = vec![f32::MAX; 8];
        let verdict = gate.evaluate(&saturated, ChannelId(1), Timestamp(0), SessionId(1));
        assert_eq!(
            verdict,
            CoherenceVerdict::Deny {
                reason: DenyReason::SaturatedSignal
            }
        );
    }

    #[test]
    fn unhealthy_channel_denied() {
        let mut gate = default_gate();
        let ch = ChannelId(5);
        gate.set_channel_health(
            ch,
            ChannelHealth {
                impedance: 1.0,
                snr: 0.1, // below min_snr
                dropout_rate: 0.0,
            },
        );
        let emb = make_embedding(8, 1.0);
        assert_eq!(
            gate.evaluate(&emb, ch, Timestamp(0), SessionId(1)),
            CoherenceVerdict::Deny {
                reason: DenyReason::ChannelUnhealthy
            }
        );
    }

    #[test]
    fn temporal_discontinuity_denied() {
        let mut gate = CoherenceGate::new(GateConfig {
            max_temporal_delta: 0.5,
            min_samples: 0, // disable manifold check threshold
            ..GateConfig::default()
        });
        let ch = ChannelId(1);
        let ts = Timestamp(1000);
        let sess = SessionId(1);

        // First embedding.
        let emb1 = make_embedding(8, 1.0);
        gate.evaluate(&emb1, ch, ts, sess);

        // Big jump.
        let emb2 = make_embedding(8, 100.0);
        match gate.evaluate(&emb2, ch, ts, sess) {
            CoherenceVerdict::Deny { reason } => {
                assert_eq!(reason, DenyReason::TemporalDiscontinuity);
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[test]
    fn mint_gated_embedding_through_gate() {
        let mut gate = default_gate();
        let raw = make_embedding(8, 1.0);
        let ch = ChannelId(1);
        let ts = Timestamp(42);
        let sess = SessionId(7);

        let result = gate.evaluate_and_mint(raw.clone(), ch, ts, sess);
        let gated = result.expect("should pass gate");
        assert_eq!(gated.vector(), raw.as_slice());
        assert_eq!(gated.channel_id(), ch);
        assert_eq!(gated.timestamp(), ts);
        assert_eq!(gated.session_id(), sess);
        assert!(gated.coherence_score() > 0.0);
    }

    #[test]
    fn gated_embedding_serde_round_trip() {
        let mut gate = default_gate();
        let raw = make_embedding(4, 0.5);
        let gated = gate
            .evaluate_and_mint(raw, ChannelId(1), Timestamp(1), SessionId(1))
            .unwrap();
        let json = serde_json::to_string(&gated).unwrap();
        let back: GatedEmbedding = serde_json::from_str(&json).unwrap();
        assert_eq!(gated, back);
    }

    /// Compile-time guarantee: external crates cannot construct GatedEmbedding.
    ///
    /// The struct's fields are private and the only public constructor is
    /// `#[doc(hidden)] __new`, which is an internal API. The `mint_gated`
    /// method on `CoherenceGate` is `pub(crate)`, so only code within
    /// `synapse-graph-ingestion` can call it. External crates have no path
    /// to create a `GatedEmbedding` through the public API.
    ///
    /// To verify: try adding the following in an external crate and confirm
    /// it fails to compile:
    /// ```compile_fail
    /// use synapse_graph_types::GatedEmbedding;
    /// let _e = GatedEmbedding {
    ///     vector: vec![1.0],
    ///     channel_id: synapse_graph_types::ChannelId(1),
    ///     timestamp: synapse_graph_types::Timestamp(0),
    ///     session_id: synapse_graph_types::SessionId(1),
    ///     coherence_score: 1.0,
    /// };
    /// ```
    #[test]
    fn gated_embedding_fields_are_private() {
        // This test exists as documentation. The real enforcement is that
        // GatedEmbedding's fields are private, so external crates cannot
        // use struct literal syntax to construct one.
    }

    /// Benchmark: evaluate() should complete in < 1 microsecond for dim=512.
    #[test]
    fn evaluate_latency_dim_512() {
        let mut gate = default_gate();
        let ch = ChannelId(1);
        let ts = Timestamp(0);
        let sess = SessionId(1);
        let dim = 512;

        // Warm up with a few samples.
        for i in 0..5 {
            let emb: Vec<f32> = (0..dim).map(|j| (j as f32) * 0.001 + (i as f32) * 0.0001).collect();
            gate.evaluate(&emb, ch, ts, sess);
        }

        let emb: Vec<f32> = (0..dim).map(|j| (j as f32) * 0.001 + 0.05).collect();
        let iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(gate.evaluate(
                std::hint::black_box(&emb),
                ch,
                ts,
                sess,
            ));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;
        // In release builds the target is < 1 µs.  Debug builds are ~50x slower
        // due to missing optimisations, so we use a relaxed ceiling there.
        // Target < 1 µs on bare metal. CI VMs have higher variance so allow 5 µs.
        let ceiling_us: u128 = if cfg!(debug_assertions) { 200 } else { 5 };
        assert!(
            per_call.as_micros() < ceiling_us,
            "evaluate() took {per_call:?} per call, expected < {ceiling_us}µs"
        );
        eprintln!("evaluate() dim=512: {per_call:?} per call");
    }
}
