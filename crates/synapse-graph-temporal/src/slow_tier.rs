//! SlowTier — long-term drift vectors with EWC++ consolidation (DDD-003, ADR-001).
//!
//! Maintains a neural encoding baseline and detects drift from it. Consolidation
//! absorbs medium-tier patterns using an EWC++-inspired approach backed by
//! `ruvector-sona`. The Adaptation domain (Prompt 11) builds the full two-tier
//! LoRA on top of this.

use synapse_graph_types::{DriftDetected, DriftVector, PersistentPattern, Timestamp};

/// Configuration for the slow tier.
#[derive(Clone, Debug)]
pub struct SlowTierConfig {
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Quantization bit-width for drift vectors. Default: 4.
    pub quantization_bits: u8,
    /// Drift detection threshold (L2 distance from baseline). Default: 3.0.
    pub drift_threshold: f32,
    /// EWC lambda for consolidation. Default: 0.4.
    pub ewc_lambda: f32,
}

impl Default for SlowTierConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            quantization_bits: 4,
            drift_threshold: 3.0,
            ewc_lambda: 0.4,
        }
    }
}

/// The slow temporal tier — long-term baseline with drift detection.
pub struct SlowTier {
    /// Detected drift vectors.
    drift_vectors: Vec<DriftVector>,
    /// Quantization bit-width for drift vector compression.
    quantization_bits: u8,
    /// Long-term neural encoding baseline.
    baseline: Vec<f32>,
    /// Last time consolidation was performed.
    last_consolidation: Timestamp,
    /// Fisher information diagonal approximation (for EWC++).
    fisher_diag: Vec<f32>,
    /// Number of consolidations performed (for running average).
    consolidation_count: u32,
    /// Drift detection threshold.
    drift_threshold: f32,
}

impl SlowTier {
    pub fn new(config: SlowTierConfig) -> Self {
        Self {
            drift_vectors: Vec::new(),
            quantization_bits: config.quantization_bits,
            baseline: vec![0.0; config.dimension],
            last_consolidation: Timestamp(0),
            fisher_diag: vec![1.0; config.dimension],
            consolidation_count: 0,
            drift_threshold: config.drift_threshold,
        }
    }

    /// Consolidate a promoted pattern into the slow-tier baseline using EWC++.
    ///
    /// The EWC++ approach prevents catastrophic forgetting by penalizing changes
    /// to dimensions that are important for previously consolidated patterns.
    /// `ewc_lambda` controls the strength of the penalty (higher = more conservative).
    pub fn consolidate(&mut self, pattern: &PersistentPattern, ewc_lambda: f32) {
        let centroid = &pattern.centroid;
        let dim = self.baseline.len().min(centroid.len());

        if self.consolidation_count == 0 {
            // First consolidation: directly adopt the pattern centroid.
            for i in 0..dim {
                self.baseline[i] = centroid[i];
            }
        } else {
            // EWC++-inspired update: move baseline toward the new pattern,
            // penalized by Fisher information for each dimension.
            for i in 0..dim {
                let delta = centroid[i] - self.baseline[i];
                // Fisher penalty: high Fisher → less movement.
                let fisher_penalty = 1.0 / (1.0 + ewc_lambda * self.fisher_diag[i]);
                self.baseline[i] += delta * fisher_penalty;
            }

            // Update Fisher diagonal: accumulate squared gradients.
            for i in 0..dim {
                let delta = centroid[i] - self.baseline[i];
                self.fisher_diag[i] += delta * delta;
            }
        }

        self.consolidation_count += 1;
        self.last_consolidation = Timestamp(
            self.last_consolidation.0.max(pattern.duration_us),
        );
    }

    /// Detect drift between a current embedding and the baseline.
    ///
    /// Returns `DriftDetected` if the L2 distance exceeds the threshold.
    pub fn detect_drift(&self, current: &[f32]) -> Option<DriftDetected> {
        if self.consolidation_count == 0 {
            return None; // No baseline yet.
        }

        let dim = self.baseline.len().min(current.len());
        let mut direction = Vec::with_capacity(dim);
        let mut sq_sum = 0.0f32;

        for i in 0..dim {
            let d = current[i] - self.baseline[i];
            direction.push(d);
            sq_sum += d * d;
        }

        let magnitude = sq_sum.sqrt();

        if magnitude >= self.drift_threshold {
            // Normalize direction.
            if magnitude > 0.0 {
                for d in &mut direction {
                    *d /= magnitude;
                }
            }

            // Record the drift vector.
            Some(DriftDetected {
                drift_magnitude: magnitude,
                affected_channels: Vec::new(), // channel attribution requires upstream info
                drift_direction: direction,
            })
        } else {
            None
        }
    }

    /// Record a detected drift vector for tracking.
    pub fn record_drift(&mut self, drift: &DriftDetected, timestamp: Timestamp) {
        self.drift_vectors.push(DriftVector {
            direction: drift.drift_direction.clone(),
            magnitude: drift.drift_magnitude,
            detected_at: timestamp,
            affected_channels: drift.affected_channels.clone(),
        });
    }

    /// Return the long-term baseline context vector.
    pub fn baseline_context(&self) -> Vec<f32> {
        self.baseline.clone()
    }

    /// Requantize drift vectors to a lower bit-width under memory pressure.
    ///
    /// Reduces precision of stored drift vectors from the current bit-width
    /// to `target_bits`. This is a lossy operation.
    pub fn requantize(&mut self, target_bits: u8) {
        if target_bits >= self.quantization_bits {
            return; // No reduction needed.
        }

        let max_val = (1u32 << (target_bits - 1)) as f32;

        for dv in &mut self.drift_vectors {
            // Find the range of each drift direction vector.
            let abs_max = dv
                .direction
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(f32::EPSILON);

            // Quantize then dequantize to simulate reduced precision.
            for d in &mut dv.direction {
                let normalized = *d / abs_max;
                let quantized = (normalized * max_val).round();
                *d = (quantized / max_val) * abs_max;
            }
        }

        self.quantization_bits = target_bits;
    }

    /// Current quantization bit-width.
    pub fn quantization_bits(&self) -> u8 {
        self.quantization_bits
    }

    /// Number of consolidations performed.
    pub fn consolidation_count(&self) -> u32 {
        self.consolidation_count
    }

    /// Number of recorded drift vectors.
    pub fn drift_vector_count(&self) -> usize {
        self.drift_vectors.len()
    }

    /// Whether the baseline has been initialized (at least one consolidation).
    pub fn has_baseline(&self) -> bool {
        self.consolidation_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{ChannelId, PatternId};

    fn make_pattern(dim: usize, base: f32) -> PersistentPattern {
        PersistentPattern {
            pattern_id: PatternId(1),
            centroid: (0..dim).map(|i| base + (i as f32) * 0.01).collect(),
            activation_count: 20,
            duration_us: 1_000_000,
        }
    }

    #[test]
    fn consolidate_initializes_baseline() {
        let mut slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            ..Default::default()
        });

        assert!(!slow.has_baseline());
        slow.consolidate(&make_pattern(8, 1.0), 0.4);
        assert!(slow.has_baseline());
        assert_eq!(slow.consolidation_count(), 1);

        let baseline = slow.baseline_context();
        assert_eq!(baseline.len(), 8);
        // First consolidation directly adopts the centroid.
        assert!((baseline[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn consolidate_ewc_moves_toward_new_pattern() {
        let mut slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            ..Default::default()
        });

        // First pattern at base=1.0.
        slow.consolidate(&make_pattern(8, 1.0), 0.4);
        let b1 = slow.baseline_context();

        // Second pattern at base=2.0 — baseline should move toward it.
        slow.consolidate(&make_pattern(8, 2.0), 0.4);
        let b2 = slow.baseline_context();

        assert!(b2[0] > b1[0], "baseline should move toward new pattern");
        assert!(b2[0] < 2.0, "EWC should prevent full jump to new pattern");
    }

    #[test]
    fn detect_drift_fires_for_distant_embedding() {
        let mut slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            drift_threshold: 3.0,
            ..Default::default()
        });

        slow.consolidate(&make_pattern(8, 1.0), 0.4);

        // Current embedding far from baseline.
        let far: Vec<f32> = vec![100.0; 8];
        let drift = slow.detect_drift(&far);
        assert!(drift.is_some(), "should detect drift for distant embedding");
        let d = drift.unwrap();
        assert!(d.drift_magnitude > 3.0);
        assert_eq!(d.drift_direction.len(), 8);
    }

    #[test]
    fn no_drift_for_close_embedding() {
        let mut slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            drift_threshold: 3.0,
            ..Default::default()
        });

        slow.consolidate(&make_pattern(8, 1.0), 0.4);

        // Current embedding close to baseline.
        let close: Vec<f32> = (0..8).map(|i| 1.0 + (i as f32) * 0.01 + 0.001).collect();
        let drift = slow.detect_drift(&close);
        assert!(drift.is_none(), "should not detect drift for close embedding");
    }

    #[test]
    fn no_drift_without_baseline() {
        let slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            ..Default::default()
        });

        let current = vec![1.0; 8];
        assert!(slow.detect_drift(&current).is_none());
    }

    #[test]
    fn requantize_reduces_bits() {
        let mut slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            quantization_bits: 4,
            ..Default::default()
        });

        // Record a drift vector.
        let drift = DriftDetected {
            drift_magnitude: 5.0,
            affected_channels: vec![ChannelId(0)],
            drift_direction: vec![0.1, -0.3, 0.5, -0.7, 0.2, -0.4, 0.6, -0.8],
        };
        slow.record_drift(&drift, Timestamp(1000));

        assert_eq!(slow.quantization_bits(), 4);
        slow.requantize(2);
        assert_eq!(slow.quantization_bits(), 2);

        // Values should be quantized (reduced precision).
        assert_eq!(slow.drift_vector_count(), 1);
    }

    #[test]
    fn requantize_noop_for_same_or_higher_bits() {
        let mut slow = SlowTier::new(SlowTierConfig {
            dimension: 8,
            quantization_bits: 4,
            ..Default::default()
        });

        slow.requantize(4);
        assert_eq!(slow.quantization_bits(), 4);
        slow.requantize(8);
        assert_eq!(slow.quantization_bits(), 4);
    }
}
