use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{ClusterId, PatternId, PersistentPattern, TemporalTier};

/// An ordered temporal sequence between two clusters (DDD-003 / DDD-004).
///
/// Consumed by the Cognitive Graph domain to create TEMPORAL_SEQUENCE edges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalSequence {
    pub from_cluster: ClusterId,
    pub to_cluster: ClusterId,
    pub confidence: f32,
    pub observed_count: u32,
    /// Average delay between the two clusters in microseconds.
    pub average_delay_us: u64,
}

/// A candidate for tier promotion (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionCandidate {
    pub pattern: PersistentPattern,
    pub from_tier: TemporalTier,
    pub to_tier: TemporalTier,
}

/// Published when a pattern is promoted between temporal tiers (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternPromoted {
    pub pattern_id: PatternId,
    pub from_tier: TemporalTier,
    pub to_tier: TemporalTier,
    pub persistence_duration_us: u64,
    pub activation_count: u32,
}

/// Compressed representation of a past session tensor (8-bit quantized).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressedSessionTensor {
    /// Quantized centroid (scaled i8 values stored as Vec<i8>).
    pub quantized_data: Vec<i8>,
    /// Scale factor for dequantization.
    pub scale: f32,
    /// Zero-point offset.
    pub zero_point: f32,
    /// Dimension of the original embedding space.
    pub dimension: usize,
    /// Number of embeddings that contributed to this tensor.
    pub sample_count: u32,
}
