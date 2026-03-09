extern crate alloc;
use alloc::vec;

use super::*;

fn round_trip<T>(val: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + core::fmt::Debug + PartialEq,
{
    let json = serde_json::to_string(val).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(*val, back);
}

#[test]
fn id_round_trips() {
    round_trip(&EmbeddingId(1));
    round_trip(&ChannelId(2));
    round_trip(&SessionId(3));
    round_trip(&NodeId(4));
    round_trip(&ClusterId(5));
    round_trip(&ShortcutId(6));
    round_trip(&PredictionId(7));
    round_trip(&EventId(8));
    round_trip(&PatternId(9));
    round_trip(&IntentId(10));
    round_trip(&LayerId(11));
    round_trip(&CollectionId(12));
}

#[test]
fn timestamp_round_trip() {
    round_trip(&Timestamp(1_700_000_000_000_000));
}

#[test]
fn spatial_position_round_trip() {
    round_trip(&SpatialPosition {
        x: 1.0,
        y: -2.5,
        z: 3.14,
    });
}

#[test]
fn temporal_tier_round_trip() {
    round_trip(&TemporalTier::Fast);
    round_trip(&TemporalTier::Medium);
    round_trip(&TemporalTier::Slow);
}

#[test]
fn temporal_meta_round_trip() {
    round_trip(&TemporalMeta {
        tier: TemporalTier::Fast,
        timestamp: Timestamp(100),
        channel_id: ChannelId(1),
        session_id: SessionId(2),
        coherence_score: 0.95,
    });
}

#[test]
fn spike_event_round_trip() {
    round_trip(&SpikeEvent {
        channel_id: ChannelId(42),
        amplitude: 0.75,
        sub_timestamp: Timestamp(999),
    });
}

#[test]
fn spike_batch_round_trip() {
    round_trip(&SpikeBatch {
        timestamp: Timestamp(1000),
        session_id: SessionId(1),
        spikes: vec![
            SpikeEvent {
                channel_id: ChannelId(1),
                amplitude: 0.5,
                sub_timestamp: Timestamp(1001),
            },
            SpikeEvent {
                channel_id: ChannelId(2),
                amplitude: 0.8,
                sub_timestamp: Timestamp(1002),
            },
        ],
        duration_us: 500,
    });
}

#[test]
fn deny_reason_round_trip() {
    round_trip(&DenyReason::ManifoldOutlier);
    round_trip(&DenyReason::TemporalDiscontinuity);
    round_trip(&DenyReason::ChannelUnhealthy);
    round_trip(&DenyReason::SaturatedSignal);
    round_trip(&DenyReason::ZeroSignal);
}

#[test]
fn coherence_verdict_round_trip() {
    round_trip(&CoherenceVerdict::Permit { score: 0.92 });
    round_trip(&CoherenceVerdict::Deny {
        reason: DenyReason::ZeroSignal,
    });
}

#[test]
fn id_types_are_copy_eq_hash() {
    use core::hash::{Hash, Hasher};

    struct NoOpHasher(u64);
    impl Hasher for NoOpHasher {
        fn finish(&self) -> u64 { self.0 }
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes { self.0 = self.0.wrapping_mul(31).wrapping_add(b as u64); }
        }
    }

    let id = NodeId(42);
    let id2: NodeId = id; // Copy
    assert_eq!(id, id2);  // Eq

    let mut h = NoOpHasher(0);
    id.hash(&mut h);
    assert_ne!(h.finish(), 0); // Hash
}

#[test]
fn channel_health_round_trip() {
    round_trip(&ChannelHealth {
        impedance: 1.5,
        snr: 12.0,
        dropout_rate: 0.02,
    });
}

#[test]
fn gated_embedding_round_trip() {
    let ge = GatedEmbedding::__new(
        vec![0.1, 0.2, 0.3],
        ChannelId(1),
        Timestamp(999),
        SessionId(2),
        0.95,
    );
    round_trip(&ge);
    // Verify getters
    assert_eq!(ge.vector(), &[0.1, 0.2, 0.3]);
    assert_eq!(ge.channel_id(), ChannelId(1));
    assert_eq!(ge.timestamp(), Timestamp(999));
    assert_eq!(ge.session_id(), SessionId(2));
    assert!((ge.coherence_score() - 0.95).abs() < f32::EPSILON);
}

#[test]
fn domain_events_round_trip() {
    round_trip(&EmbeddingDenied {
        embedding_id: EmbeddingId(1),
        reason: DenyReason::ManifoldOutlier,
        channel_id: ChannelId(2),
        timestamp: Timestamp(100),
    });

    round_trip(&ChannelHealthChanged {
        channel_id: ChannelId(3),
        previous: ChannelHealth {
            impedance: 1.0,
            snr: 10.0,
            dropout_rate: 0.01,
        },
        current: ChannelHealth {
            impedance: 1.2,
            snr: 8.0,
            dropout_rate: 0.05,
        },
    });

    let ge = GatedEmbedding::__new(
        vec![0.5],
        ChannelId(1),
        Timestamp(0),
        SessionId(1),
        0.9,
    );
    round_trip(&EmbeddingIngested {
        embedding_id: EmbeddingId(42),
        gated_embedding: ge,
    });
}
