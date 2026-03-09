//! CausalResolver — determines predecessor EventIds for NeuralEvents (DDD-007).
//!
//! Given a NeuralEvent and the current DAG state, resolves which prior events
//! are causal predecessors of the new event.

use synapse_graph_types::{EventId, EventType, NeuralEvent};

use crate::dag_store::DagStore;

/// Resolves causal predecessors for events based on domain rules.
pub struct CausalResolver;

impl CausalResolver {
    pub fn new() -> Self {
        Self
    }

    /// Determine predecessor EventIds for a given NeuralEvent.
    ///
    /// Resolution rules per DDD-007:
    /// - Ingested, Denied, ChannelHealth → [] (root events)
    /// - Stored → [corresponding Ingested event]
    /// - Evicted → [] (root event)
    /// - Promoted → [recent Ingested/Stored events]
    /// - TierEvicted → [] (root event)
    /// - Drift → [] (root event)
    /// - ShortcutFound → [recent Promoted + Stored events]
    /// - Healed → [most recent TopologyChanged event]
    /// - ProofFail → [] (root event)
    /// - TopologyChanged → [] (root event)
    /// - Predicted → [recent ShortcutFound events]
    /// - NewIntent → [] (root event)
    /// - AccuracyDrop → [recent Predicted events]
    /// - Adapted → [the Predicted event that triggered it]
    /// - Consolidated → [recent Adapted events]
    /// - Oscillation → [recent Adapted events]
    /// - FisherUpdated → [] (root event)
    pub fn resolve(&self, event: &NeuralEvent, dag: &DagStore) -> Vec<EventId> {
        match event {
            // Root events: no predecessors.
            NeuralEvent::Ingested(_)
            | NeuralEvent::Denied(_)
            | NeuralEvent::ChannelHealth(_)
            | NeuralEvent::Evicted(_)
            | NeuralEvent::TierEvicted(_)
            | NeuralEvent::Drift(_)
            | NeuralEvent::ProofFail(_)
            | NeuralEvent::TopologyChanged(_)
            | NeuralEvent::NewIntent(_)
            | NeuralEvent::FisherUpdated(_) => Vec::new(),

            // Stored → most recent Ingested event.
            NeuralEvent::Stored(_) => {
                self.find_recent_by_type(dag, EventType::Ingestion, 1)
            }

            // Promoted → recent Ingested + Stored events.
            NeuralEvent::Promoted(_) => {
                let mut preds = self.find_recent_by_type(dag, EventType::Ingestion, 3);
                preds.extend(self.find_recent_by_type(dag, EventType::Storage, 3));
                preds
            }

            // ShortcutFound → recent Promoted + Stored events.
            NeuralEvent::ShortcutFound(_) => {
                let mut preds = self.find_recent_by_type(dag, EventType::TierPromotion, 3);
                preds.extend(self.find_recent_by_type(dag, EventType::Storage, 3));
                preds
            }

            // Healed → most recent TopologyChanged event.
            NeuralEvent::Healed(_) => {
                self.find_recent_by_type(dag, EventType::SystemHealth, 1)
            }

            // Predicted → recent ShortcutFound events.
            NeuralEvent::Predicted(_) => {
                self.find_recent_by_type(dag, EventType::ShortcutDiscovery, 5)
            }

            // AccuracyDrop → recent Predicted events.
            NeuralEvent::AccuracyDrop(_) => {
                self.find_recent_by_type(dag, EventType::Prediction, 3)
            }

            // Adapted → most recent Predicted event.
            NeuralEvent::Adapted(_) => {
                self.find_recent_by_type(dag, EventType::Prediction, 1)
            }

            // Consolidated → recent Adapted events.
            NeuralEvent::Consolidated(_) => {
                self.find_recent_by_type(dag, EventType::Adaptation, 10)
            }

            // Oscillation → recent Adapted events.
            NeuralEvent::Oscillation(_) => {
                self.find_recent_by_type(dag, EventType::Adaptation, 5)
            }
        }
    }

    /// Find the most recent N events of a given type in the DAG.
    fn find_recent_by_type(
        &self,
        dag: &DagStore,
        event_type: EventType,
        max: usize,
    ) -> Vec<EventId> {
        dag.iter()
            .filter(|node| node.event_type == event_type)
            .collect::<Vec<_>>()
            .iter()
            .rev()
            .take(max)
            .map(|node| node.id)
            .collect()
    }
}

impl Default for CausalResolver {
    fn default() -> Self {
        Self::new()
    }
}
