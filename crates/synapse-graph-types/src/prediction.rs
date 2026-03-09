use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::{CognitiveShortcut, IntentId, PredictionId, ShortcutId, Timestamp};

/// A shortcut matched by the ShortcutMatcher (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchedShortcut {
    pub shortcut: CognitiveShortcut,
    /// Distance (or 1 - similarity) between the partial pattern and the shortcut.
    pub distance: f32,
}

/// A scored intent produced by the AttentionScorer (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredIntent {
    pub intent_id: IntentId,
    pub confidence: f32,
    pub shortcut_id: ShortcutId,
    pub match_distance: f32,
}

/// An entry in the intent vocabulary (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntentEntry {
    pub id: IntentId,
    pub label: String,
    pub shortcut_ids: Vec<ShortcutId>,
    pub activation_count: u64,
    pub accuracy_rate: f32,
    pub first_seen: Timestamp,
    pub last_predicted: Timestamp,
    /// Rolling window of recent correct/incorrect outcomes.
    pub recent_outcomes: Vec<bool>,
}

/// The final prediction output (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntentPrediction {
    pub id: PredictionId,
    pub timestamp: Timestamp,
    pub ranked_intents: Vec<ScoredIntent>,
    pub attention_weights: Vec<f32>,
    pub temporal_influence: f32,
    pub latency_us: u64,
}

/// Feedback on a prediction (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PredictionFeedback {
    /// The prediction was correct.
    Confirmed,
    /// The prediction was wrong; the actual intent is provided.
    Corrected { actual: IntentId },
    /// The prediction was rejected by the user.
    Rejected,
    /// No feedback received within the window.
    Timeout,
}

/// Signal sent to the Adaptation domain when prediction accuracy degrades (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptationSignal {
    pub signal_type: AdaptationSignalType,
    pub urgency: AdaptationUrgency,
    pub prediction_id: PredictionId,
    pub details: String,
}

/// Type of adaptation signal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationSignalType {
    PredictionError,
    AccuracyDegraded,
}

/// Urgency level for adaptation signals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationUrgency {
    Low,
    Medium,
    High,
}

// -----------------------------------------------------------------------
// Domain events
// -----------------------------------------------------------------------

/// Published when a prediction is made (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionMade {
    pub prediction: IntentPrediction,
    pub feedback_window_ms: u64,
}

/// Published when a new intent is discovered during cold start (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewIntentDiscovered {
    pub intent_id: IntentId,
    pub initial_pattern: Vec<f32>,
    pub discovery_context: String,
}

/// Published when rolling accuracy drops below threshold (DDD-005).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccuracyDegraded {
    pub current_accuracy: f32,
    pub threshold: f32,
    pub window_size: usize,
}
