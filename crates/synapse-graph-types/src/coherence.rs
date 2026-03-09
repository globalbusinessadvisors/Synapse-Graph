use serde::{Deserialize, Serialize};

/// Reason a spike batch was denied by the coherence gate (DDD-001).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DenyReason {
    ManifoldOutlier,
    TemporalDiscontinuity,
    ChannelUnhealthy,
    SaturatedSignal,
    ZeroSignal,
}

/// Result of coherence gating (ADR-002).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CoherenceVerdict {
    Permit { score: f32 },
    Deny { reason: DenyReason },
}
