//! Adaptation domain types (DDD-006, ADR-004).
//!
//! Types for the two-tier LoRA adaptation engine: fast path (rank-4) for
//! immediate corrections and slow path (rank-16) for consolidated learning.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{LayerId, Timestamp};

/// What triggered an adaptation delta.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Fast path: immediate prediction error correction.
    FastCorrection,
    /// Slow path: consolidated EWC++ learning.
    SlowConsolidation,
}

/// Summary of a LoRA delta for diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeltaSummary {
    pub layer_id: LayerId,
    pub rank: usize,
    pub memory_bytes: usize,
    pub trigger: AdaptationType,
    pub computed_at: Timestamp,
}

/// A low-rank adaptation delta (DDD-006).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraDelta {
    pub layer_id: LayerId,
    /// Shape: (d_out, rank) stored row-major.
    pub matrix_a: Vec<f32>,
    /// Shape: (d_in, rank) stored row-major.
    pub matrix_b: Vec<f32>,
    pub rank: usize,
    pub computed_at: Timestamp,
    pub trigger: AdaptationType,
}

impl LoraDelta {
    /// Total memory consumed by the delta matrices in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.matrix_a.len() + self.matrix_b.len()) * core::mem::size_of::<f32>()
    }

    /// Produce a diagnostic summary.
    pub fn summary(&self) -> DeltaSummary {
        DeltaSummary {
            layer_id: self.layer_id,
            rank: self.rank,
            memory_bytes: self.memory_bytes(),
            trigger: self.trigger.clone(),
            computed_at: self.computed_at,
        }
    }
}

/// Which adaptation path was used.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationPath {
    Fast,
    Slow,
    Queued,
}

/// Result of a single adapt() call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptationResult {
    pub delta: Option<LoraDelta>,
    pub latency_us: u64,
    pub path: AdaptationPath,
    pub ewc_penalty: f32,
}

/// Result of a consolidation cycle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub deltas_consolidated: usize,
    pub total_latency_us: u64,
    pub ewc_penalty: f32,
    pub weights_protected: usize,
    pub fisher_recomputed: bool,
}

/// Configuration for the AdaptationEngine.
#[derive(Clone, Debug)]
pub struct AdaptationConfig {
    /// Rank for fast LoRA deltas. Default: 4.
    pub fast_rank: usize,
    /// Rank for slow LoRA deltas. Default: 16.
    pub slow_rank: usize,
    /// EWC lambda for slow path. Default: 5000.0.
    pub ewc_lambda: f32,
    /// Damping factor for fast path EMA. Default: 0.9.
    pub damping_factor: f32,
    /// Maximum accumulated fast deltas before auto-consolidation. Default: 100.
    pub max_accumulated: usize,
    /// Target layers for fast path (attention heads + classification).
    pub fast_target_layers: Vec<LayerId>,
    /// Target layers for slow path (all layers).
    pub slow_target_layers: Vec<LayerId>,
    /// Oscillation window size. Default: 10.
    pub oscillation_window: usize,
    /// Number of sign flips to declare oscillation. Default: 6.
    pub oscillation_flip_threshold: usize,
    /// Fisher staleness threshold (microseconds). Default: 60_000_000 (60s).
    pub fisher_staleness_us: u64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            fast_rank: 4,
            slow_rank: 16,
            ewc_lambda: 5000.0,
            damping_factor: 0.9,
            max_accumulated: 100,
            fast_target_layers: vec![LayerId(0), LayerId(1)],
            slow_target_layers: vec![LayerId(0), LayerId(1), LayerId(2), LayerId(3)],
            oscillation_window: 10,
            oscillation_flip_threshold: 6,
            fisher_staleness_us: 60_000_000,
        }
    }
}

/// Error type for LoRA delta application.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoraError {
    /// The target does not have the specified layer.
    LayerNotFound(LayerId),
    /// Matrix dimensions are incompatible.
    DimensionMismatch { expected: usize, got: usize },
    /// Application failed for another reason.
    ApplicationFailed(String),
}

/// Trait for targets that can receive LoRA delta updates (DDD-006).
///
/// Implemented by GnnEngine (slow deltas) and AttentionScorer (fast deltas).
pub trait LoraTarget {
    /// Apply a LoRA delta to this target.
    fn apply_lora_delta(&mut self, delta: &LoraDelta) -> Result<(), LoraError>;
    /// Return the layer IDs this target manages.
    fn layer_ids(&self) -> Vec<LayerId>;
}

// -----------------------------------------------------------------------
// Domain events (DDD-006)
// -----------------------------------------------------------------------

/// Published when an adaptation delta is applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptationApplied {
    pub result: AdaptationResult,
    pub affected_layers: Vec<LayerId>,
    pub trigger: AdaptationType,
}

/// Published when fast deltas are consolidated into the slow path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Consolidated {
    pub result: ConsolidationResult,
    pub accumulated_fast_deltas_count: usize,
}

/// Published when oscillation is detected on a layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OscillationDetected {
    pub layer_id: LayerId,
    pub magnitude: f32,
    pub damping_applied: f32,
}

/// Published when the Fisher information matrix is recomputed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FisherRecomputed {
    pub sample_count: u64,
    pub top_protected_weights: Vec<usize>,
}
