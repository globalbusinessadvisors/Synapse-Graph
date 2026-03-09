use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{ChannelId, SessionId, Timestamp};

/// A single spike event on a channel (DDD-001).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub channel_id: ChannelId,
    pub amplitude: f32,
    pub sub_timestamp: Timestamp,
}

/// A batch of spike events within a time window (DDD-001).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpikeBatch {
    pub timestamp: Timestamp,
    pub session_id: SessionId,
    pub spikes: Vec<SpikeEvent>,
    pub duration_us: u32,
}
