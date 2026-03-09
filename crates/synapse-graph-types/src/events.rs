use serde::{Deserialize, Serialize};

use crate::{ChannelHealth, ChannelId, DenyReason, EmbeddingId, GatedEmbedding, Timestamp};

/// Published when an embedding passes the CoherenceGate (DDD-001).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingIngested {
    pub embedding_id: EmbeddingId,
    pub gated_embedding: GatedEmbedding,
}

/// Published when an embedding is rejected by the CoherenceGate (DDD-001).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingDenied {
    pub embedding_id: EmbeddingId,
    pub reason: DenyReason,
    pub channel_id: ChannelId,
    pub timestamp: Timestamp,
}

/// Published when a channel's health metrics change (DDD-001).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelHealthChanged {
    pub channel_id: ChannelId,
    pub previous: ChannelHealth,
    pub current: ChannelHealth,
}
