use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{
    ChannelId, CollectionId, EmbeddingId, SessionId, TemporalMeta, TemporalTier, Timestamp,
};

/// Result of a k-nearest-neighbor search (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub embedding_id: EmbeddingId,
    pub vector: Vec<f32>,
    pub distance: f32,
    pub metadata: TemporalMeta,
}

/// A cluster of nearby embeddings found by HNSW neighborhood analysis (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCluster {
    pub centroid: Vec<f32>,
    pub member_ids: Vec<EmbeddingId>,
    pub metadata: Vec<TemporalMeta>,
    /// Temporal span: (earliest, latest) timestamps among members.
    pub temporal_span: (Timestamp, Timestamp),
}

/// A temporally adjacent embedding from another cluster (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalNeighbor {
    pub embedding_id: EmbeddingId,
    pub correlation: f32,
}

/// Filter criteria for a collection (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollectionFilter {
    pub session_id: Option<SessionId>,
    pub channel_range: Option<(ChannelId, ChannelId)>,
    pub temporal_tier: Option<TemporalTier>,
}

/// A named logical partition of the vector store (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Collection {
    pub id: CollectionId,
    pub name: String,
    pub filter: CollectionFilter,
    pub embedding_count: usize,
    pub created_at: Timestamp,
}

/// Error on insertion into VectorMemory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsertError {
    /// The embedding dimension does not match the store's configured dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// Duplicate embedding ID.
    DuplicateId(EmbeddingId),
}

impl core::fmt::Display for InsertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::DuplicateId(id) => write!(f, "duplicate embedding id: {}", id.0),
        }
    }
}

/// Reason embeddings were evicted from a collection (DDD-002).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvictionReason {
    MemoryPressure,
    TierExpiry,
    ManualPrune,
}

/// Published when an embedding is stored in VectorMemory (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingStored {
    pub embedding_id: EmbeddingId,
    pub collection_id: Option<CollectionId>,
    pub metadata: TemporalMeta,
    /// IDs of the embedding's nearest HNSW graph neighbors at insert time.
    pub hnsw_neighbors: Vec<EmbeddingId>,
}

/// Published when embeddings are evicted (DDD-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsEvicted {
    pub collection_id: Option<CollectionId>,
    pub count: usize,
    pub reason: EvictionReason,
}
