//! FastTier — sub-10 ms spike-burst buffer with pattern detection (DDD-003, ADR-001).
//!
//! Maintains a fixed-capacity ring buffer of recent embeddings and monitors for
//! patterns that persist long enough to be promoted to the medium tier.

use synapse_graph_types::{
    EvictionTrigger, GatedEmbedding, PatternId, PersistentPattern, TemporalTier, TierEvicted,
    Timestamp, TimestampedEmbedding,
};

use crate::ring_buffer::RingBuffer;

/// Configuration for the fast tier.
#[derive(Clone, Debug)]
pub struct FastTierConfig {
    /// Maximum number of embeddings held in the ring buffer.
    pub capacity: usize,
    /// Minimum persistence (µs) before a pattern is considered for promotion.
    /// Default: 500_000 µs (500 ms).
    pub min_persistence_us: u64,
    /// Minimum number of activations for promotion candidacy. Default: 10.
    pub min_activations: u32,
    /// L2 distance threshold for grouping embeddings into the same pattern.
    pub pattern_distance_threshold: f32,
}

impl Default for FastTierConfig {
    fn default() -> Self {
        Self {
            capacity: 1024,
            min_persistence_us: 500_000,
            min_activations: 10,
            pattern_distance_threshold: 2.0,
        }
    }
}

/// Tracks an evolving burst pattern.
#[derive(Clone, Debug)]
struct PatternTracker {
    id: PatternId,
    centroid: Vec<f32>,
    count: u32,
    first_seen: Timestamp,
    last_seen: Timestamp,
    member_count: u32, // for running centroid average
}

/// Detects burst patterns that persist beyond the fast tier's natural window.
#[derive(Clone, Debug)]
struct BurstPatternDetector {
    patterns: Vec<PatternTracker>,
    next_id: u64,
    threshold: f32,
}

impl BurstPatternDetector {
    fn new(threshold: f32) -> Self {
        Self {
            patterns: Vec::new(),
            next_id: 1,
            threshold,
        }
    }

    /// Record an embedding observation. Updates an existing pattern tracker or
    /// creates a new one.
    fn observe(&mut self, vector: &[f32], timestamp: Timestamp) {
        // Find the nearest existing pattern.
        let mut best_idx = None;
        let mut best_dist = f32::MAX;
        for (i, pat) in self.patterns.iter().enumerate() {
            let dist = l2_distance(vector, &pat.centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if best_dist <= self.threshold {
            // Update existing pattern.
            let pat = &mut self.patterns[best_idx.unwrap()];
            pat.count += 1;
            pat.last_seen = timestamp;
            // Running centroid update.
            pat.member_count += 1;
            let n = pat.member_count as f32;
            for (j, v) in vector.iter().enumerate() {
                if j < pat.centroid.len() {
                    pat.centroid[j] += (v - pat.centroid[j]) / n;
                }
            }
        } else {
            // New pattern.
            let id = PatternId(self.next_id);
            self.next_id += 1;
            self.patterns.push(PatternTracker {
                id,
                centroid: vector.to_vec(),
                count: 1,
                first_seen: timestamp,
                last_seen: timestamp,
                member_count: 1,
            });
        }
    }

    /// Return patterns that exceed the persistence and activation thresholds.
    fn persistent_patterns(
        &self,
        min_persistence_us: u64,
        min_activations: u32,
    ) -> Vec<PersistentPattern> {
        self.patterns
            .iter()
            .filter_map(|pat| {
                let duration_us = pat.last_seen.0.saturating_sub(pat.first_seen.0);
                if duration_us >= min_persistence_us && pat.count >= min_activations {
                    Some(PersistentPattern {
                        pattern_id: pat.id,
                        centroid: pat.centroid.clone(),
                        activation_count: pat.count,
                        duration_us,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

/// The fast temporal tier — ring buffer of recent spike-burst embeddings.
pub struct FastTier {
    buffer: RingBuffer<TimestampedEmbedding>,
    config: FastTierConfig,
    pattern_detector: BurstPatternDetector,
    /// Cumulative eviction events.
    total_evicted: usize,
}

impl FastTier {
    pub fn new(config: FastTierConfig) -> Self {
        let pattern_detector =
            BurstPatternDetector::new(config.pattern_distance_threshold);
        let buffer = RingBuffer::new(config.capacity);
        Self {
            buffer,
            config,
            pattern_detector,
            total_evicted: 0,
        }
    }

    /// Push an embedding into the fast tier buffer.
    ///
    /// Returns a [`TierEvicted`] event if the oldest entry was overwritten.
    pub fn push(
        &mut self,
        embedding: &GatedEmbedding,
        timestamp: Timestamp,
    ) -> Option<TierEvicted> {
        self.pattern_detector
            .observe(embedding.vector(), timestamp);

        let evicted = self.buffer.push(TimestampedEmbedding {
            embedding: embedding.clone(),
            timestamp,
        });

        if evicted {
            self.total_evicted += 1;
            Some(TierEvicted {
                tier: TemporalTier::Fast,
                evicted_count: 1,
                reason: EvictionTrigger::RingBufferOverflow,
            })
        } else {
            None
        }
    }

    /// Detect patterns that have persisted beyond the fast tier window and are
    /// candidates for promotion to the medium tier.
    pub fn detect_persistent_patterns(&self) -> Vec<PersistentPattern> {
        self.pattern_detector
            .persistent_patterns(self.config.min_persistence_us, self.config.min_activations)
    }

    /// Return embeddings from the recent time window (in microseconds).
    pub fn recent_context(&self, window_us: u64) -> Vec<&GatedEmbedding> {
        let now = self
            .buffer
            .iter()
            .last()
            .map(|te| te.timestamp.0)
            .unwrap_or(0);
        let cutoff = now.saturating_sub(window_us);
        self.buffer
            .iter()
            .filter(|te| te.timestamp.0 >= cutoff)
            .map(|te| &te.embedding)
            .collect()
    }

    /// Current number of embeddings in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Total number of embeddings evicted since creation.
    pub fn total_evicted(&self) -> usize {
        self.total_evicted
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

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{ChannelId, SessionId};

    fn make_gated(base: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..8).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), 0.9)
    }

    #[test]
    fn ring_buffer_evicts_oldest_at_capacity() {
        let config = FastTierConfig {
            capacity: 3,
            ..FastTierConfig::default()
        };
        let mut tier = FastTier::new(config);
        assert!(tier.push(&make_gated(1.0), Timestamp(100)).is_none());
        assert!(tier.push(&make_gated(2.0), Timestamp(200)).is_none());
        assert!(tier.push(&make_gated(3.0), Timestamp(300)).is_none());
        // 4th push evicts the oldest (1.0).
        let evt = tier.push(&make_gated(4.0), Timestamp(400));
        assert!(evt.is_some());
        let evt = evt.unwrap();
        assert_eq!(evt.tier, TemporalTier::Fast);
        assert_eq!(evt.reason, EvictionTrigger::RingBufferOverflow);
        assert_eq!(tier.len(), 3);
        assert_eq!(tier.total_evicted(), 1);

        // Buffer should contain 2.0, 3.0, 4.0 (oldest first).
        let ctx = tier.recent_context(u64::MAX);
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn detect_persistent_pattern_planted() {
        let config = FastTierConfig {
            capacity: 256,
            min_persistence_us: 500_000, // 500ms
            min_activations: 10,
            pattern_distance_threshold: 1.0,
            ..FastTierConfig::default()
        };
        let mut tier = FastTier::new(config);

        // Plant a persistent pattern: same-ish embedding over > 500ms with > 10 activations.
        for i in 0..15u64 {
            let base = 1.0 + (i as f32) * 0.001; // small variance
            let ts = Timestamp(i * 50_000); // 50ms apart → 15 × 50ms = 750ms total
            tier.push(&make_gated(base), ts);
        }

        let patterns = tier.detect_persistent_patterns();
        assert!(
            !patterns.is_empty(),
            "expected at least one persistent pattern"
        );
        let pat = &patterns[0];
        assert!(pat.activation_count >= 10);
        assert!(pat.duration_us >= 500_000);
    }

    #[test]
    fn no_persistent_patterns_for_short_bursts() {
        let config = FastTierConfig {
            capacity: 64,
            min_persistence_us: 500_000,
            min_activations: 10,
            pattern_distance_threshold: 1.0,
            ..FastTierConfig::default()
        };
        let mut tier = FastTier::new(config);

        // Short burst: only 3 activations over 100ms.
        for i in 0..3u64 {
            tier.push(&make_gated(1.0), Timestamp(i * 30_000));
        }

        let patterns = tier.detect_persistent_patterns();
        assert!(patterns.is_empty(), "should find no persistent patterns");
    }

    #[test]
    fn recent_context_window() {
        let config = FastTierConfig {
            capacity: 64,
            ..FastTierConfig::default()
        };
        let mut tier = FastTier::new(config);

        tier.push(&make_gated(1.0), Timestamp(100));
        tier.push(&make_gated(2.0), Timestamp(200));
        tier.push(&make_gated(3.0), Timestamp(500));
        tier.push(&make_gated(4.0), Timestamp(1000));

        // Window of 500µs from latest (1000): should include timestamps >= 500.
        let ctx = tier.recent_context(500);
        assert_eq!(ctx.len(), 2); // timestamps 500 and 1000
    }

    #[test]
    fn push_latency_benchmark() {
        let config = FastTierConfig {
            capacity: 4096,
            ..FastTierConfig::default()
        };
        let mut tier = FastTier::new(config);
        let emb = make_gated(1.0);

        let iterations = 10_000u32;
        let start = std::time::Instant::now();
        for i in 0..iterations {
            std::hint::black_box(tier.push(
                std::hint::black_box(&emb),
                Timestamp(i as u64 * 100),
            ));
        }
        let per_call = start.elapsed() / iterations;
        let ceiling_us: u128 = if cfg!(debug_assertions) { 50 } else { 1 };
        assert!(
            per_call.as_micros() < ceiling_us,
            "push() took {per_call:?}, expected < {ceiling_us}µs"
        );
        eprintln!("push(): {per_call:?}");
    }
}
