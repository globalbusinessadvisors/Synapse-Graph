use serde::{Deserialize, Serialize};

use crate::{ChannelId, SessionId, Timestamp};

/// Three-tier temporal resolution (ADR-001).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalTier {
    Fast,
    Medium,
    Slow,
}

/// Metadata attached to temporally-aware domain objects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalMeta {
    pub tier: TemporalTier,
    pub timestamp: Timestamp,
    pub channel_id: ChannelId,
    pub session_id: SessionId,
    pub coherence_score: f32,
}
