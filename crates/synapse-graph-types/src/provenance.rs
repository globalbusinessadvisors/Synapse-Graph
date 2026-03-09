//! Provenance domain types (DDD-007).
//!
//! NeuralEvent union enum and supporting types for the provenance DAG.

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{
    // DDD-001 events
    EmbeddingIngested, EmbeddingDenied, ChannelHealthChanged,
    // DDD-002 events
    EmbeddingStored, EmbeddingsEvicted,
    // DDD-003 events
    PatternPromoted, TierEvicted, DriftDetected,
    // DDD-004 events
    ShortcutDiscovered, GraphHealed, ProofFailure, GraphTopologyChanged,
    // DDD-005 events
    PredictionMade, NewIntentDiscovered, AccuracyDegraded,
    // DDD-006 events
    AdaptationApplied, Consolidated, OscillationDetected, FisherRecomputed,
    // IDs
    EventId, LayerId, SessionId, ShortcutId, Timestamp,
};

/// Union of all domain events across all bounded contexts (DDD-007).
///
/// This is the terminal event type consumed by the ProvenanceLog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NeuralEvent {
    // DDD-001: Spike Ingestion
    Ingested(EmbeddingIngested),
    Denied(EmbeddingDenied),
    ChannelHealth(ChannelHealthChanged),

    // DDD-002: Vector Memory
    Stored(EmbeddingStored),
    Evicted(EmbeddingsEvicted),

    // DDD-003: Temporal Learning
    Promoted(PatternPromoted),
    TierEvicted(TierEvicted),
    Drift(DriftDetected),

    // DDD-004: Cognitive Graph
    ShortcutFound(ShortcutDiscovered),
    Healed(GraphHealed),
    ProofFail(ProofFailure),
    TopologyChanged(GraphTopologyChanged),

    // DDD-005: Intent Prediction
    Predicted(PredictionMade),
    NewIntent(NewIntentDiscovered),
    AccuracyDrop(AccuracyDegraded),

    // DDD-006: Adaptation
    Adapted(AdaptationApplied),
    Consolidated(Consolidated),
    Oscillation(OscillationDetected),
    FisherUpdated(FisherRecomputed),
}

impl NeuralEvent {
    /// Return the EventType classification for this event.
    pub fn event_type(&self) -> EventType {
        match self {
            NeuralEvent::Ingested(_) => EventType::Ingestion,
            NeuralEvent::Denied(_) => EventType::Denial,
            NeuralEvent::ChannelHealth(_) => EventType::SystemHealth,
            NeuralEvent::Stored(_) => EventType::Storage,
            NeuralEvent::Evicted(_) => EventType::Eviction,
            NeuralEvent::Promoted(_) => EventType::TierPromotion,
            NeuralEvent::TierEvicted(_) => EventType::Eviction,
            NeuralEvent::Drift(_) => EventType::DriftDetection,
            NeuralEvent::ShortcutFound(_) => EventType::ShortcutDiscovery,
            NeuralEvent::Healed(_) => EventType::GraphHealing,
            NeuralEvent::ProofFail(_) => EventType::ProofFailure,
            NeuralEvent::TopologyChanged(_) => EventType::SystemHealth,
            NeuralEvent::Predicted(_) => EventType::Prediction,
            NeuralEvent::NewIntent(_) => EventType::Prediction,
            NeuralEvent::AccuracyDrop(_) => EventType::SystemHealth,
            NeuralEvent::Adapted(_) => EventType::Adaptation,
            NeuralEvent::Consolidated(_) => EventType::Consolidation,
            NeuralEvent::Oscillation(_) => EventType::Adaptation,
            NeuralEvent::FisherUpdated(_) => EventType::SystemHealth,
        }
    }
}

/// Classification of event types for the provenance DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    Ingestion,
    Denial,
    Storage,
    Eviction,
    TierPromotion,
    DriftDetection,
    ShortcutDiscovery,
    GraphHealing,
    ProofFailure,
    Prediction,
    Adaptation,
    Consolidation,
    SystemHealth,
}

/// A node in the provenance DAG (DDD-007).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventNode {
    pub id: EventId,
    pub timestamp: Timestamp,
    pub session_id: SessionId,
    pub event_type: EventType,
    /// Serialized NeuralEvent payload.
    pub payload: Vec<u8>,
    /// IDs of predecessor events in the causal chain.
    pub predecessor_ids: Vec<EventId>,
    /// SHA-256(payload || predecessor_0.hash || ... || predecessor_n.hash).
    pub hash: [u8; 32],
}

/// An audit trail produced by tracing a causal chain (DDD-007).
#[derive(Clone, Debug, PartialEq)]
pub struct AuditTrail {
    pub target_event: EventId,
    pub causal_chain: Vec<EventId>,
    pub depth: usize,
    pub domains_involved: Vec<EventType>,
}

/// Result of a DAG integrity verification (DDD-007).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub total_events: u64,
    pub verified_events: u64,
    pub corrupted_events: Vec<EventId>,
    pub is_valid: bool,
}

/// Query parameters for provenance lookups.
#[derive(Clone, Debug)]
pub struct ProvenanceQuery {
    pub session_id: Option<SessionId>,
    pub shortcut_id: Option<ShortcutId>,
    pub layer_id: Option<LayerId>,
    pub event_type: Option<EventType>,
}
