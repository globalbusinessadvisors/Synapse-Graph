use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{ChannelId, SessionId, Timestamp};

/// An embedding that has passed the CoherenceGate (ADR-002).
///
/// All fields are private. The only way to construct this type is through
/// `CoherenceGate::mint_gated` in `synapse-graph-ingestion`, which enforces
/// the hard architectural boundary: no raw embedding can reach downstream
/// modules without coherence verification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatedEmbedding {
    vector: Vec<f32>,
    channel_id: ChannelId,
    timestamp: Timestamp,
    session_id: SessionId,
    coherence_score: f32,
}

impl GatedEmbedding {
    /// Internal constructor. Only available to crates that depend on
    /// `synapse-graph-types` and use the `_internal_constructor` feature.
    #[doc(hidden)]
    pub fn __new(
        vector: Vec<f32>,
        channel_id: ChannelId,
        timestamp: Timestamp,
        session_id: SessionId,
        coherence_score: f32,
    ) -> Self {
        Self {
            vector,
            channel_id,
            timestamp,
            session_id,
            coherence_score,
        }
    }

    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    pub fn channel_id(&self) -> ChannelId {
        self.channel_id
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub fn coherence_score(&self) -> f32 {
        self.coherence_score
    }
}
