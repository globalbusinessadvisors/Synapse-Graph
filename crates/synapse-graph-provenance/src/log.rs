//! ProvenanceLog — aggregate root for provenance tracking (DDD-007).

use synapse_graph_types::{IntegrityReport, EventId, NeuralEvent, SessionId};

use crate::dag_store::DagStore;
use crate::query::ProvenanceQueryEngine;
use crate::resolver::CausalResolver;

/// The ProvenanceLog aggregate root (DDD-007).
///
/// Terminal event sink that consumes domain events from all bounded contexts
/// and stores them in a tamper-evident DAG with automatic causal resolution.
pub struct ProvenanceLog {
    dag: DagStore,
    resolver: CausalResolver,
    current_session: SessionId,
}

impl ProvenanceLog {
    pub fn new(session: SessionId) -> Self {
        Self {
            dag: DagStore::new(),
            resolver: CausalResolver::new(),
            current_session: session,
        }
    }

    /// Append a NeuralEvent to the provenance DAG.
    ///
    /// Uses the CausalResolver to automatically determine predecessor events,
    /// then delegates to the DagStore.
    pub fn append(&mut self, event: NeuralEvent) -> EventId {
        let predecessors = self.resolver.resolve(&event, &self.dag);
        self.dag.append(event, self.current_session, predecessors)
    }

    /// Create a query engine for the provenance DAG.
    pub fn query(&self) -> ProvenanceQueryEngine<'_> {
        ProvenanceQueryEngine::new(&self.dag)
    }

    /// Verify the integrity of the entire DAG.
    pub fn verify(&self) -> IntegrityReport {
        self.query().verify_integrity()
    }

    /// Set the current session for subsequent events.
    pub fn set_session(&mut self, session: SessionId) {
        self.current_session = session;
    }

    /// Access the underlying DAG store.
    pub fn dag(&self) -> &DagStore {
        &self.dag
    }

    /// Mutable access to the DAG store (for testing).
    pub fn dag_mut(&mut self) -> &mut DagStore {
        &mut self.dag
    }

    /// Total number of events in the log.
    pub fn total_events(&self) -> u64 {
        self.dag.total_events()
    }
}
