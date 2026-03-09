//! MediumTier — session-level tensor accumulation with 8-bit quantization
//! (DDD-003, ADR-001).
//!
//! Accumulates gated embeddings into a running session tensor, compresses
//! completed sessions into 8-bit quantized archives, and extracts temporal
//! sequences for the Cognitive Graph domain.

use std::collections::HashMap;

use synapse_graph_types::{
    ClusterId, CompressedSessionTensor, EmbeddingCluster, GatedEmbedding,
    PersistentPattern, TemporalSequence,
};

/// Configuration for the medium tier.
#[derive(Clone, Debug)]
pub struct MediumTierConfig {
    /// Quantization bit-width for compressed sessions. Default: 8.
    pub quantization_bits: u8,
    /// Maximum number of archived past sessions. Default: 10.
    pub max_past_sessions: usize,
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Minimum number of sequence observations for confidence. Default: 3.
    pub min_sequence_observations: u32,
}

impl Default for MediumTierConfig {
    fn default() -> Self {
        Self {
            quantization_bits: 8,
            max_past_sessions: 10,
            dimension: 128,
            min_sequence_observations: 3,
        }
    }
}

/// Running statistics for the current session.
#[derive(Clone, Debug)]
pub struct SessionTensor {
    /// Running mean of accumulated embeddings.
    mean: Vec<f32>,
    /// Running M2 (for variance) per dimension.
    m2: Vec<f32>,
    /// Number of accumulated embeddings.
    count: u32,
    /// Ordered cluster assignments for sequence extraction.
    cluster_trace: Vec<(ClusterId, u64)>, // (cluster_id, timestamp_us)
}

impl SessionTensor {
    fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            count: 0,
            cluster_trace: Vec::new(),
        }
    }

    fn accumulate(&mut self, vector: &[f32]) {
        self.count += 1;
        let n = self.count as f32;
        let inv_n = 1.0 / n;
        for i in 0..vector.len().min(self.mean.len()) {
            let delta = vector[i] - self.mean[i];
            self.mean[i] += delta * inv_n;
            let delta2 = vector[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    fn record_cluster(&mut self, cluster_id: ClusterId, timestamp_us: u64) {
        self.cluster_trace.push((cluster_id, timestamp_us));
    }

    /// Compress into an 8-bit quantized tensor.
    fn compress(&self) -> CompressedSessionTensor {
        if self.count == 0 {
            return CompressedSessionTensor {
                quantized_data: vec![0i8; self.mean.len()],
                scale: 1.0,
                zero_point: 0.0,
                dimension: self.mean.len(),
                sample_count: 0,
            };
        }

        let min_val = self.mean.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = self.mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(f32::EPSILON);
        let scale = range / 254.0; // 8-bit range: -127..127
        let zero_point = min_val;

        let quantized_data: Vec<i8> = self
            .mean
            .iter()
            .map(|&v| {
                let normalized = (v - zero_point) / scale;
                (normalized - 127.0).clamp(-128.0, 127.0) as i8
            })
            .collect();

        CompressedSessionTensor {
            quantized_data,
            scale,
            zero_point,
            dimension: self.mean.len(),
            sample_count: self.count,
        }
    }
}

/// The medium temporal tier — session-level tensor accumulation.
pub struct MediumTier {
    current_session: SessionTensor,
    past_sessions: Vec<CompressedSessionTensor>,
    config: MediumTierConfig,
}

impl MediumTier {
    pub fn new(config: MediumTierConfig) -> Self {
        let dim = config.dimension;
        Self {
            current_session: SessionTensor::new(dim),
            past_sessions: Vec::new(),
            config,
        }
    }

    /// Accumulate a gated embedding into the current session tensor.
    pub fn accumulate(&mut self, embedding: &GatedEmbedding) {
        self.current_session.accumulate(embedding.vector());
    }

    /// Record a cluster assignment for sequence extraction.
    pub fn record_cluster(&mut self, cluster_id: ClusterId, timestamp_us: u64) {
        self.current_session.record_cluster(cluster_id, timestamp_us);
    }

    /// Extract temporal sequences from the current session's cluster trace.
    ///
    /// Examines ordered cluster pairs that appear consistently, returning
    /// [`TemporalSequence`] values for the Cognitive Graph domain.
    pub fn extract_sequences(&self, _cluster: &EmbeddingCluster) -> Vec<TemporalSequence> {
        self.extract_all_sequences()
    }

    /// Extract all temporal sequences from the current session.
    pub fn extract_all_sequences(&self) -> Vec<TemporalSequence> {
        let trace = &self.current_session.cluster_trace;
        if trace.len() < 2 {
            return Vec::new();
        }

        // Count ordered pair transitions and accumulate delays.
        let mut pair_stats: HashMap<(ClusterId, ClusterId), (u32, u64)> = HashMap::new();
        for window in trace.windows(2) {
            let (from_id, from_ts) = window[0];
            let (to_id, to_ts) = window[1];
            if from_id != to_id {
                let entry = pair_stats.entry((from_id, to_id)).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += to_ts.saturating_sub(from_ts);
            }
        }

        let total_transitions = pair_stats.values().map(|(c, _)| *c as f32).sum::<f32>().max(1.0);

        pair_stats
            .iter()
            .filter(|(_, (count, _))| *count >= self.config.min_sequence_observations)
            .map(|(&(from, to), &(count, total_delay))| {
                TemporalSequence {
                    from_cluster: from,
                    to_cluster: to,
                    confidence: count as f32 / total_transitions,
                    observed_count: count,
                    average_delay_us: total_delay / count as u64,
                }
            })
            .collect()
    }

    /// Return the current session's context vector (running mean).
    pub fn session_context(&self) -> Vec<f32> {
        self.current_session.mean.clone()
    }

    /// End the current session: compress and archive, start fresh.
    pub fn end_session(&mut self) {
        if self.current_session.count > 0 {
            let compressed = self.current_session.compress();
            self.past_sessions.push(compressed);

            // Evict oldest if over capacity.
            while self.past_sessions.len() > self.config.max_past_sessions {
                self.past_sessions.remove(0);
            }
        }

        self.current_session = SessionTensor::new(self.config.dimension);
    }

    /// Compute cross-session stability of a pattern.
    ///
    /// Returns a score in [0.0, 1.0] indicating how consistently the pattern's
    /// centroid appears across archived sessions. Used by TierPromoter to decide
    /// medium-to-slow promotion.
    pub fn cross_session_stability(&self, pattern: &PersistentPattern) -> f32 {
        if self.past_sessions.is_empty() {
            return 0.0;
        }

        let mut matching_sessions = 0u32;
        for session in &self.past_sessions {
            let dequantized = dequantize(session);
            let dist = l2_distance(&pattern.centroid, &dequantized);
            // Consider a match if the pattern centroid is within reasonable distance
            // of the session mean.
            if dist < 5.0 {
                matching_sessions += 1;
            }
        }

        matching_sessions as f32 / self.past_sessions.len() as f32
    }

    /// Number of archived past sessions.
    pub fn past_session_count(&self) -> usize {
        self.past_sessions.len()
    }

    /// Number of embeddings in the current session.
    pub fn current_session_count(&self) -> u32 {
        self.current_session.count
    }
}

fn dequantize(tensor: &CompressedSessionTensor) -> Vec<f32> {
    tensor
        .quantized_data
        .iter()
        .map(|&q| (q as f32 + 127.0) * tensor.scale + tensor.zero_point)
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{ChannelId, SessionId, Timestamp};

    fn make_gated(dim: usize, base: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..dim).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), 0.9)
    }

    #[test]
    fn accumulate_produces_nonzero_context() {
        let mut tier = MediumTier::new(MediumTierConfig {
            dimension: 8,
            ..Default::default()
        });

        for i in 0..100 {
            tier.accumulate(&make_gated(8, 1.0 + (i as f32) * 0.01));
        }

        let ctx = tier.session_context();
        assert_eq!(ctx.len(), 8);
        assert!(ctx.iter().any(|&v| v != 0.0), "context should be non-zero");
        assert_eq!(tier.current_session_count(), 100);
    }

    #[test]
    fn extract_sequences_finds_planted_pattern() {
        let mut tier = MediumTier::new(MediumTierConfig {
            dimension: 8,
            min_sequence_observations: 3,
            ..Default::default()
        });

        let cluster_a = ClusterId(1);
        let cluster_b = ClusterId(2);

        // Plant an A→B pattern: repeated transitions.
        for i in 0..10u64 {
            tier.record_cluster(cluster_a, i * 1000);
            tier.record_cluster(cluster_b, i * 1000 + 500);
        }

        let dummy_cluster = EmbeddingCluster {
            centroid: vec![0.0; 8],
            member_ids: vec![],
            metadata: vec![],
            temporal_span: (Timestamp(0), Timestamp(10000)),
        };
        let sequences = tier.extract_sequences(&dummy_cluster);
        assert!(
            !sequences.is_empty(),
            "expected at least one temporal sequence"
        );

        let ab = sequences
            .iter()
            .find(|s| s.from_cluster == cluster_a && s.to_cluster == cluster_b);
        assert!(ab.is_some(), "expected A→B sequence");
        let ab = ab.unwrap();
        assert!(ab.observed_count >= 3);
        assert!(ab.confidence > 0.0);
        assert_eq!(ab.average_delay_us, 500);
    }

    #[test]
    fn end_session_compresses_and_archives() {
        let mut tier = MediumTier::new(MediumTierConfig {
            dimension: 8,
            max_past_sessions: 3,
            ..Default::default()
        });

        // Accumulate and end 4 sessions.
        for session in 0..4 {
            for i in 0..10 {
                let base = (session as f32) * 1.0 + (i as f32) * 0.01;
                tier.accumulate(&make_gated(8, base));
            }
            tier.end_session();
        }

        // max_past_sessions=3 so oldest should be evicted.
        assert_eq!(tier.past_session_count(), 3);
        // Current session should be fresh.
        assert_eq!(tier.current_session_count(), 0);

        // Verify compressed tensors have correct structure.
        for session in &tier.past_sessions {
            assert_eq!(session.dimension, 8);
            assert_eq!(session.quantized_data.len(), 8);
            assert!(session.sample_count > 0);
        }
    }

    #[test]
    fn cross_session_stability() {
        let mut tier = MediumTier::new(MediumTierConfig {
            dimension: 8,
            max_past_sessions: 10,
            ..Default::default()
        });

        // Create 5 sessions with similar centroids around base=1.0.
        for _ in 0..5 {
            for i in 0..20 {
                tier.accumulate(&make_gated(8, 1.0 + (i as f32) * 0.001));
            }
            tier.end_session();
        }

        // Pattern near 1.0 should have high stability.
        let stable_pattern = PersistentPattern {
            pattern_id: synapse_graph_types::PatternId(1),
            centroid: (0..8).map(|i| 1.0 + (i as f32) * 0.01).collect(),
            activation_count: 20,
            duration_us: 1_000_000,
        };
        let stability = tier.cross_session_stability(&stable_pattern);
        assert!(
            stability > 0.5,
            "stable pattern should have high stability, got {stability}"
        );

        // Pattern far away should have low stability.
        let unstable_pattern = PersistentPattern {
            pattern_id: synapse_graph_types::PatternId(2),
            centroid: vec![100.0; 8],
            activation_count: 5,
            duration_us: 500_000,
        };
        let stability = tier.cross_session_stability(&unstable_pattern);
        assert!(
            stability < 0.5,
            "distant pattern should have low stability, got {stability}"
        );
    }

    #[test]
    fn accumulate_latency_benchmark() {
        let mut tier = MediumTier::new(MediumTierConfig {
            dimension: 128,
            ..Default::default()
        });
        let emb = make_gated(128, 1.0);

        let iterations = 10_000u32;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(&mut tier).accumulate(std::hint::black_box(&emb));
        }
        let per_call = start.elapsed() / iterations;
        let ceiling_us: u128 = if cfg!(debug_assertions) { 100 } else { 50 };
        assert!(
            per_call.as_micros() < ceiling_us,
            "accumulate() took {per_call:?}, expected < {ceiling_us}µs"
        );
        eprintln!("accumulate() dim=128: {per_call:?}");
    }

    #[test]
    fn extract_sequences_latency_benchmark() {
        let mut tier = MediumTier::new(MediumTierConfig {
            dimension: 8,
            min_sequence_observations: 2,
            ..Default::default()
        });

        // Build a substantial cluster trace.
        for i in 0..1000u64 {
            let cluster = ClusterId(i % 20);
            tier.record_cluster(cluster, i * 100);
        }

        let dummy = EmbeddingCluster {
            centroid: vec![0.0; 8],
            member_ids: vec![],
            metadata: vec![],
            temporal_span: (Timestamp(0), Timestamp(100000)),
        };

        let iterations = 100u32;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(tier.extract_sequences(std::hint::black_box(&dummy)));
        }
        let per_call = start.elapsed() / iterations;
        let ceiling_ms: u128 = if cfg!(debug_assertions) { 50 } else { 5 };
        assert!(
            per_call.as_millis() < ceiling_ms,
            "extract_sequences() took {per_call:?}, expected < {ceiling_ms}ms"
        );
        eprintln!("extract_sequences(): {per_call:?}");
    }
}
