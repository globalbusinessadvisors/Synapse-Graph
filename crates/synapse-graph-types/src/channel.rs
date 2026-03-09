use serde::{Deserialize, Serialize};

/// Health metrics for a single input channel (DDD-001).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelHealth {
    pub impedance: f32,
    pub snr: f32,
    pub dropout_rate: f32,
}
