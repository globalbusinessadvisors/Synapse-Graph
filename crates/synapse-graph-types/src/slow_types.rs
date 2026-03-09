use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{ChannelId, Timestamp};

/// A detected drift direction in the slow tier's baseline encoding (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftVector {
    /// Unit direction of the drift in embedding space.
    pub direction: Vec<f32>,
    /// Magnitude of the drift.
    pub magnitude: f32,
    /// When the drift was detected.
    pub detected_at: Timestamp,
    /// Which channels contributed to the drift.
    pub affected_channels: Vec<ChannelId>,
}

/// Published when significant drift is detected from the slow-tier baseline (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftDetected {
    pub drift_magnitude: f32,
    pub affected_channels: Vec<ChannelId>,
    pub drift_direction: Vec<f32>,
}

/// Configurable weights for cross-tier context blending (DDD-003).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierWeights {
    pub fast: f32,
    pub medium: f32,
    pub slow: f32,
}

impl Default for TierWeights {
    fn default() -> Self {
        Self {
            fast: 0.5,
            medium: 0.3,
            slow: 0.2,
        }
    }
}

/// Cross-tier blended temporal context (DDD-003).
///
/// Consumed by the Intent Prediction domain (Prompt 10).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalContext {
    pub fast_component: Vec<f32>,
    pub medium_component: Vec<f32>,
    pub slow_component: Vec<f32>,
    pub weights: TierWeights,
}

impl TemporalContext {
    /// Blend all three tier components into a single context vector using the
    /// configured weights.
    pub fn blended(&self) -> Vec<f32> {
        let len = self
            .fast_component
            .len()
            .max(self.medium_component.len())
            .max(self.slow_component.len());

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let f = self.fast_component.get(i).copied().unwrap_or(0.0) * self.weights.fast;
            let m = self.medium_component.get(i).copied().unwrap_or(0.0) * self.weights.medium;
            let s = self.slow_component.get(i).copied().unwrap_or(0.0) * self.weights.slow;
            result.push(f + m + s);
        }
        result
    }
}

/// Result of a maintenance cycle on the TemporalLearner (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceResult {
    /// Number of patterns promoted during this cycle.
    pub promoted: usize,
    /// Number of embeddings evicted during this cycle.
    pub evicted: usize,
}
