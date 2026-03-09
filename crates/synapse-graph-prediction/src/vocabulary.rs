//! IntentVocabulary — manages known intents and their accuracy (DDD-005).

use std::collections::HashMap;

use synapse_graph_types::{AccuracyDegraded, IntentEntry, IntentId, ShortcutId, Timestamp};

/// Configuration for the intent vocabulary.
#[derive(Clone, Debug)]
pub struct VocabularyConfig {
    /// Rolling window size for accuracy tracking. Default: 100.
    pub accuracy_window: usize,
    /// Accuracy threshold below which AccuracyDegraded fires. Default: 0.7.
    pub accuracy_threshold: f32,
}

impl Default for VocabularyConfig {
    fn default() -> Self {
        Self {
            accuracy_window: 100,
            accuracy_threshold: 0.7,
        }
    }
}

/// Manages the set of known intents.
pub struct IntentVocabulary {
    intents: HashMap<IntentId, IntentEntry>,
    next_id: u64,
    config: VocabularyConfig,
}

impl IntentVocabulary {
    pub fn new(config: VocabularyConfig) -> Self {
        Self {
            intents: HashMap::new(),
            next_id: 1,
            config,
        }
    }

    /// Register a new intent with a label.
    pub fn register(&mut self, label: String) -> IntentId {
        let id = IntentId(self.next_id);
        self.next_id += 1;

        let now = Timestamp(0);
        self.intents.insert(
            id,
            IntentEntry {
                id,
                label,
                shortcut_ids: Vec::new(),
                activation_count: 0,
                accuracy_rate: 1.0,
                first_seen: now,
                last_predicted: now,
                recent_outcomes: Vec::new(),
            },
        );
        id
    }

    /// Register with a specific ID (for testing).
    pub fn register_with_id(&mut self, id: IntentId, label: String) {
        let now = Timestamp(0);
        self.intents.insert(
            id,
            IntentEntry {
                id,
                label,
                shortcut_ids: Vec::new(),
                activation_count: 0,
                accuracy_rate: 1.0,
                first_seen: now,
                last_predicted: now,
                recent_outcomes: Vec::new(),
            },
        );
        self.next_id = self.next_id.max(id.0 + 1);
    }

    /// Get an intent entry.
    pub fn get(&self, id: &IntentId) -> Option<&IntentEntry> {
        self.intents.get(id)
    }

    /// Associate a shortcut with an intent.
    pub fn associate_shortcut(&mut self, intent_id: &IntentId, shortcut_id: ShortcutId) {
        if let Some(entry) = self.intents.get_mut(intent_id) {
            if !entry.shortcut_ids.contains(&shortcut_id) {
                entry.shortcut_ids.push(shortcut_id);
            }
        }
    }

    /// Record a prediction accuracy outcome.
    ///
    /// Returns `Some(AccuracyDegraded)` if the rolling accuracy drops below threshold.
    pub fn record_accuracy(
        &mut self,
        id: &IntentId,
        correct: bool,
    ) -> Option<AccuracyDegraded> {
        let entry = self.intents.get_mut(id)?;
        entry.activation_count += 1;

        entry.recent_outcomes.push(correct);
        if entry.recent_outcomes.len() > self.config.accuracy_window {
            entry.recent_outcomes.remove(0);
        }

        // Compute rolling accuracy.
        let total = entry.recent_outcomes.len();
        if total > 0 {
            let correct_count = entry.recent_outcomes.iter().filter(|&&v| v).count();
            entry.accuracy_rate = correct_count as f32 / total as f32;
        }

        // Check for degradation.
        if total >= 10 && entry.accuracy_rate < self.config.accuracy_threshold {
            Some(AccuracyDegraded {
                current_accuracy: entry.accuracy_rate,
                threshold: self.config.accuracy_threshold,
                window_size: total,
            })
        } else {
            None
        }
    }

    /// Record a prediction timestamp.
    pub fn record_prediction(&mut self, id: &IntentId, timestamp: Timestamp) {
        if let Some(entry) = self.intents.get_mut(id) {
            entry.last_predicted = timestamp;
        }
    }

    /// Number of registered intents.
    pub fn count(&self) -> usize {
        self.intents.len()
    }

    /// Check if an intent exists.
    pub fn contains(&self, id: &IntentId) -> bool {
        self.intents.contains_key(id)
    }
}

impl Default for IntentVocabulary {
    fn default() -> Self {
        Self::new(VocabularyConfig::default())
    }
}
