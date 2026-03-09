use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{GatedEmbedding, PatternId, TemporalTier, Timestamp};

/// A gated embedding with its capture timestamp, stored in temporal tier buffers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimestampedEmbedding {
    pub embedding: GatedEmbedding,
    pub timestamp: Timestamp,
}

/// A pattern that has persisted long enough to be a candidate for tier promotion
/// (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistentPattern {
    pub pattern_id: PatternId,
    pub centroid: Vec<f32>,
    pub activation_count: u32,
    /// Duration of pattern persistence in microseconds.
    pub duration_us: u64,
}

/// Published when embeddings are evicted from a temporal tier buffer (DDD-003).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierEvicted {
    pub tier: TemporalTier,
    pub evicted_count: usize,
    pub reason: EvictionTrigger,
}

/// What triggered a tier eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvictionTrigger {
    RingBufferOverflow,
    MemoryPressure,
    TierExpiry,
}
