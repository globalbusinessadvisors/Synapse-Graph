//! ProvenanceQueryEngine — query interface for the provenance DAG (DDD-007).

use std::collections::HashSet;

use synapse_graph_types::{
    AuditTrail, EventId, EventNode, EventType, IntegrityReport, LayerId, SessionId, ShortcutId,
    NeuralEvent,
};

use crate::dag_store::DagStore;

/// Query engine for the provenance DAG.
pub struct ProvenanceQueryEngine<'a> {
    dag: &'a DagStore,
}

impl<'a> ProvenanceQueryEngine<'a> {
    pub fn new(dag: &'a DagStore) -> Self {
        Self { dag }
    }

    /// Trace the causal chain backward from an event to root events.
    ///
    /// Walks predecessor_ids recursively, building the complete causal chain.
    pub fn trace(&self, event_id: EventId) -> AuditTrail {
        let mut causal_chain = Vec::new();
        let mut visited = HashSet::new();
        let mut domains = HashSet::new();
        let mut stack = vec![event_id];

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }

            if let Some(node) = self.dag.get(current) {
                causal_chain.push(current);
                domains.insert(node.event_type);

                for &pred in &node.predecessor_ids {
                    if !visited.contains(&pred) {
                        stack.push(pred);
                    }
                }
            }
        }

        let depth = causal_chain.len().saturating_sub(1);
        let domains_involved: Vec<EventType> = domains.into_iter().collect();

        AuditTrail {
            target_event: event_id,
            causal_chain,
            depth,
            domains_involved,
        }
    }

    /// Return all events from a specific session.
    pub fn session_slice(&self, session_id: SessionId) -> Vec<&EventNode> {
        self.dag
            .iter()
            .filter(|node| node.session_id == session_id)
            .collect()
    }

    /// Find the discovery event for a specific shortcut.
    pub fn shortcut_origin(&self, shortcut_id: ShortcutId) -> Option<&EventNode> {
        self.dag
            .iter()
            .filter(|node| node.event_type == EventType::ShortcutDiscovery)
            .find(|node| {
                // Deserialize payload to check shortcut ID.
                if let Ok(event) = serde_json::from_slice::<NeuralEvent>(&node.payload) {
                    if let NeuralEvent::ShortcutFound(ref disc) = event {
                        return disc.shortcut.id == shortcut_id;
                    }
                }
                false
            })
    }

    /// Find all adaptation events for a specific layer.
    pub fn adaptation_history(&self, layer_id: LayerId) -> Vec<&EventNode> {
        self.dag
            .iter()
            .filter(|node| node.event_type == EventType::Adaptation)
            .filter(|node| {
                if let Ok(event) = serde_json::from_slice::<NeuralEvent>(&node.payload) {
                    if let NeuralEvent::Adapted(ref applied) = event {
                        return applied.affected_layers.contains(&layer_id);
                    }
                }
                false
            })
            .collect()
    }

    /// Verify the integrity of the entire DAG.
    ///
    /// Recomputes every hash and checks for mismatches.
    pub fn verify_integrity(&self) -> IntegrityReport {
        let mut total = 0u64;
        let mut verified = 0u64;
        let mut corrupted = Vec::new();

        // Build a map of all nodes for hash recomputation.
        let nodes: std::collections::HashMap<_, _> = self
            .dag
            .iter()
            .map(|node| (node.id, node))
            .collect();

        let nodes_for_hash: std::collections::HashMap<_, _> = nodes
            .iter()
            .map(|(&id, &node)| (id, node.clone()))
            .collect();

        for node in self.dag.iter() {
            total += 1;

            let recomputed = DagStore::compute_hash_from(
                &node.payload,
                &node.predecessor_ids,
                &nodes_for_hash,
            );

            if recomputed == node.hash {
                verified += 1;
            } else {
                corrupted.push(node.id);
            }
        }

        IntegrityReport {
            total_events: total,
            verified_events: verified,
            corrupted_events: corrupted.clone(),
            is_valid: corrupted.is_empty(),
        }
    }
}
