//! DagStore — persistent event DAG wrapping ruvector-dag (DDD-007).
//!
//! Stores EventNodes in a tamper-evident directed acyclic graph.
//! Each event's hash covers its payload and all predecessor hashes.

use std::collections::HashMap;

use synapse_graph_types::{EventId, EventNode, NeuralEvent, SessionId, Timestamp};

/// The DAG store for provenance events.
///
/// In production, backs onto ruvector-dag v2.0.4.
/// This implementation uses an in-memory HashMap.
pub struct DagStore {
    nodes: HashMap<EventId, EventNode>,
    next_id: u64,
    /// Ordered list of event IDs for sequential iteration.
    ordered: Vec<EventId>,
}

impl DagStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 1,
            ordered: Vec::new(),
        }
    }

    /// Append a new event to the DAG.
    ///
    /// Pipeline:
    /// a) Serialize event to payload bytes.
    /// b) Compute hash: SHA-256(payload || predecessor_0.hash || ... || predecessor_n.hash).
    /// c) Create EventNode with monotonically increasing ID.
    /// d) Store in DAG.
    /// e) Return EventId.
    ///
    /// Performance target: < 5 microseconds per append.
    pub fn append(
        &mut self,
        event: NeuralEvent,
        session: SessionId,
        predecessors: Vec<EventId>,
    ) -> EventId {
        let id = EventId(self.next_id);
        self.next_id += 1;

        let event_type = event.event_type();

        // a) Serialize event to payload.
        let payload = serde_json::to_vec(&event).unwrap_or_default();

        // b) Compute hash.
        let hash = Self::compute_hash(&payload, &predecessors, &self.nodes);

        // c) Create EventNode.
        let node = EventNode {
            id,
            timestamp: Timestamp(0), // caller should set real timestamp
            session_id: session,
            event_type,
            payload,
            predecessor_ids: predecessors,
            hash,
        };

        // d) Store.
        self.nodes.insert(id, node);
        self.ordered.push(id);

        // e) Return.
        id
    }

    /// Append with a specific timestamp.
    pub fn append_with_timestamp(
        &mut self,
        event: NeuralEvent,
        session: SessionId,
        predecessors: Vec<EventId>,
        timestamp: Timestamp,
    ) -> EventId {
        let id = EventId(self.next_id);
        self.next_id += 1;

        let event_type = event.event_type();
        let payload = serde_json::to_vec(&event).unwrap_or_default();
        let hash = Self::compute_hash(&payload, &predecessors, &self.nodes);

        let node = EventNode {
            id,
            timestamp,
            session_id: session,
            event_type,
            payload,
            predecessor_ids: predecessors,
            hash,
        };

        self.nodes.insert(id, node);
        self.ordered.push(id);
        id
    }

    /// Get an event node by ID.
    pub fn get(&self, id: EventId) -> Option<&EventNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to an event node (for testing tamper scenarios).
    pub fn get_mut(&mut self, id: EventId) -> Option<&mut EventNode> {
        self.nodes.get_mut(&id)
    }

    /// Total number of events in the DAG.
    pub fn total_events(&self) -> u64 {
        self.nodes.len() as u64
    }

    /// Iterate over all event nodes in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &EventNode> {
        self.ordered.iter().filter_map(|id| self.nodes.get(id))
    }

    /// Compute the hash for an event: SHA-256(payload || predecessor hashes).
    ///
    /// Uses a simple deterministic hash since we don't have a SHA-256 dependency.
    /// In production, this would use a proper cryptographic hash.
    fn compute_hash(
        payload: &[u8],
        predecessors: &[EventId],
        nodes: &HashMap<EventId, EventNode>,
    ) -> [u8; 32] {
        Self::compute_hash_from(payload, predecessors, nodes)
    }

    /// Public version for recomputation during verification.
    pub(crate) fn compute_hash_from(
        payload: &[u8],
        predecessors: &[EventId],
        nodes: &HashMap<EventId, EventNode>,
    ) -> [u8; 32] {
        let mut hash = [0u8; 32];

        // Fold payload bytes.
        for (i, &b) in payload.iter().enumerate() {
            hash[i % 32] ^= b;
        }

        // Fold predecessor hashes.
        for pred_id in predecessors {
            if let Some(pred_node) = nodes.get(pred_id) {
                for (i, &b) in pred_node.hash.iter().enumerate() {
                    hash[(i + 11) % 32] ^= b;
                }
            }
        }

        // Mix pass for better distribution.
        for i in 0..32 {
            hash[i] = hash[i]
                .wrapping_add(hash[(i + 1) % 32].wrapping_mul(37))
                .wrapping_add(hash[(i + 7) % 32].wrapping_mul(13));
        }

        hash
    }
}

impl Default for DagStore {
    fn default() -> Self {
        Self::new()
    }
}
