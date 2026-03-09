# DDD-007: Provenance Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R10 | ADR-005, ADR-007
**Implementing Phase:** Phase 5

---

## Domain Overview

The Provenance domain maintains the tamper-evident, append-only log of all neural events, decisions, and adaptations within SynapseGraph. It provides auditability, debugging, and clinical traceability -- answering questions like "why was this intent predicted?", "when did this shortcut emerge?", and "what adaptations occurred during this session?"

Every other domain publishes events to Provenance. Provenance never modifies other domains' state.

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Provenance Log** | The append-only DAG (Directed Acyclic Graph) of all system events. Each event references its causal predecessors. |
| **Neural Event** | Any event in the SynapseGraph system: ingestion, denial, storage, prediction, adaptation, graph healing, etc. |
| **Event Node** | A node in the provenance DAG representing a single event with its data, timestamp, and hash. |
| **Causal Link** | A directed edge in the DAG from a predecessor event to a successor event, indicating that the predecessor causally influenced the successor. |
| **Hash Chain** | Each event node includes a cryptographic hash of its content and its predecessors' hashes, forming a tamper-evident chain. |
| **Provenance Query** | A query over the DAG to trace the causal history of a specific event (e.g., "what caused this prediction?"). |
| **Session Slice** | A subgraph of the provenance DAG containing all events from a single session. |
| **Audit Trail** | The complete causal chain from a specific event back to the original spike ingestion events that contributed to it. |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 PROVENANCE CONTEXT                                 |
|                                                                   |
|  Inbound (events from all other domains):                         |
|    EmbeddingIngested, EmbeddingDenied (Spike Ingestion)           |
|    EmbeddingStored, EmbeddingsEvicted (Vector Memory)             |
|    PatternPromoted, TierEvicted, DriftDetected (Temporal)         |
|    ShortcutDiscovered, GraphHealed, ProofFailure (Cognitive)      |
|    PredictionMade (Intent Prediction)                             |
|    AdaptationApplied, Consolidated (Adaptation)                   |
|                                                                   |
|  +---------------------+                                         |
|  | ProvenanceLog       |  Aggregate Root                         |
|  | (owns DAG +         |  Append-only event log with             |
|  |  query engine)      |  tamper-evident hash chain              |
|  +----------+----------+                                         |
|             |                                                     |
|    +--------+--------+                                            |
|    |                 |                                            |
|    v                 v                                            |
|  +----------+  +----------------+                                 |
|  | DAG      |  | Query Engine   |                                 |
|  | Store    |  | (trace, audit) |                                 |
|  +----------+  +----------------+                                 |
|                                                                   |
|  Outbound:                                                        |
|    Audit trails (read-only, for UI / clinical review)             |
|    Integrity reports (for verification)                           |
|                                                                   |
+------------------------------------------------------------------+
```

### Context Map

All other domains have a **Published Language** relationship with Provenance as the downstream consumer. Provenance is a pure **event sink** -- it consumes events from all domains but never produces events that other domains act on.

| Upstream Domain | Event Types |
|----------------|-------------|
| Spike Ingestion (DDD-001) | `EmbeddingIngested`, `EmbeddingDenied`, `ChannelHealthChanged` |
| Vector Memory (DDD-002) | `EmbeddingStored`, `EmbeddingsEvicted` |
| Temporal Learning (DDD-003) | `PatternPromoted`, `TierEvicted`, `DriftDetected` |
| Cognitive Graph (DDD-004) | `ShortcutDiscovered`, `GraphHealed`, `ProofFailure`, `GraphTopologyChanged` |
| Intent Prediction (DDD-005) | `PredictionMade`, `NewIntentDiscovered`, `AccuracyDegraded` |
| Adaptation (DDD-006) | `AdaptationApplied`, `Consolidated`, `OscillationDetected`, `FisherRecomputed` |

---

## Domain Model

### Aggregates

#### ProvenanceLog (Aggregate Root)

```rust
/// Aggregate root for the tamper-evident event log.
/// Invariant: the DAG is append-only; events are never modified or deleted.
/// Invariant: every event's hash includes its predecessors' hashes.
/// Invariant: append latency < 5 microseconds.
struct ProvenanceLog {
    dag: DagStore,                   // ruvector-dag
    query_engine: ProvenanceQueryEngine,
    current_session: SessionId,
    event_count: u64,
    config: ProvenanceConfig,
}

impl ProvenanceLog {
    /// Append an event to the provenance DAG.
    fn append(&mut self, event: NeuralEvent) -> EventId;

    /// Query: trace the causal chain of a specific event.
    fn trace(&self, event_id: EventId) -> AuditTrail;

    /// Query: get all events in a session.
    fn session_slice(&self, session_id: SessionId) -> Vec<EventNode>;

    /// Query: find when a specific shortcut was first discovered.
    fn shortcut_origin(&self, shortcut_id: ShortcutId) -> Option<EventNode>;

    /// Query: get all adaptation events affecting a specific layer.
    fn adaptation_history(&self, layer_id: LayerId) -> Vec<EventNode>;

    /// Verify integrity of the hash chain.
    fn verify_integrity(&self) -> IntegrityReport;
}
```

### Entities

#### EventNode

```rust
/// A node in the provenance DAG.
/// Identity: event_id (monotonically increasing).
struct EventNode {
    id: EventId,
    timestamp: Timestamp,
    session_id: SessionId,
    event_type: EventType,
    payload: EventPayload,           // serialized event data
    predecessor_ids: Vec<EventId>,   // causal predecessors
    hash: [u8; 32],                  // SHA-256 of (payload + predecessor hashes)
}
```

### Value Objects

#### NeuralEvent

```rust
/// Union type for all domain events accepted by the provenance log.
enum NeuralEvent {
    // Spike Ingestion
    Ingested(EmbeddingIngested),
    Denied(EmbeddingDenied),
    ChannelHealth(ChannelHealthChanged),

    // Vector Memory
    Stored(EmbeddingStored),
    Evicted(EmbeddingsEvicted),

    // Temporal Learning
    Promoted(PatternPromoted),
    TierEvicted(TierEvicted),
    Drift(DriftDetected),

    // Cognitive Graph
    ShortcutFound(ShortcutDiscovered),
    Healed(GraphHealed),
    ProofFail(ProofFailure),
    TopologyChanged(GraphTopologyChanged),

    // Intent Prediction
    Predicted(PredictionMade),
    NewIntent(NewIntentDiscovered),
    AccuracyDrop(AccuracyDegraded),

    // Adaptation
    Adapted(AdaptationApplied),
    Consolidated(Consolidated),
    Oscillation(OscillationDetected),
    FisherUpdated(FisherRecomputed),
}
```

#### AuditTrail

```rust
/// The complete causal chain for a specific event.
struct AuditTrail {
    target_event: EventNode,
    causal_chain: Vec<EventNode>,   // ordered from root cause to target
    depth: u32,                     // number of causal steps
    domains_involved: Vec<String>,  // which bounded contexts contributed
}
```

#### IntegrityReport

```rust
/// Result of hash chain verification.
struct IntegrityReport {
    total_events: u64,
    verified_events: u64,
    corrupted_events: Vec<EventId>,  // should always be empty
    verification_time: Duration,
    is_valid: bool,
}
```

#### EventType

```rust
/// Classification of events for indexing and querying.
enum EventType {
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
```

### Domain Events

The Provenance domain itself does not publish domain events (it is the terminal sink). However, it does produce:

```rust
/// Result of integrity verification (not a domain event, but a query result).
/// May trigger alerts in the system health monitoring layer.
struct IntegrityViolation {
    event_id: EventId,
    expected_hash: [u8; 32],
    actual_hash: [u8; 32],
    severity: Severity,
}
```

---

## Domain Services

### DagStore

**Backed by:** `ruvector-dag` v2.0.4

The persistent DAG data structure. Provides:
- O(1) append with hash chain maintenance.
- O(log N) lookup by event ID.
- O(depth) causal chain traversal.

**Performance contract:** < 5 microseconds per append.

### ProvenanceQueryEngine

Provides query capabilities over the DAG:

| Query | Description | Complexity |
|-------|-------------|------------|
| `trace(event_id)` | Full causal chain to root events | O(depth * branching_factor) |
| `session_slice(session_id)` | All events in a session | O(session_events) |
| `shortcut_origin(shortcut_id)` | When/how a shortcut was discovered | O(log N) |
| `adaptation_history(layer_id)` | All adaptations to a layer | O(adaptations) |
| `verify_integrity()` | Full hash chain verification | O(N) |

### Causal Predecessor Resolution

When an event is appended, the provenance log automatically determines its causal predecessors:

| Event Type | Causal Predecessors |
|-----------|---------------------|
| `EmbeddingIngested` | The `SpikeBatch` that produced it (session-level predecessor) |
| `EmbeddingStored` | The corresponding `EmbeddingIngested` event |
| `EmbeddingDenied` | The corresponding `SpikeBatch` event |
| `PatternPromoted` | The `EmbeddingIngested` events that formed the pattern |
| `ShortcutDiscovered` | The `PatternPromoted` and `EmbeddingStored` events in the shortcut's subgraph |
| `PredictionMade` | The `ShortcutDiscovered` events for matched shortcuts + the partial pattern's `EmbeddingIngested` |
| `AdaptationApplied` | The `PredictionMade` event that triggered adaptation |
| `Consolidated` | All `AdaptationApplied` events being consolidated |
| `GraphHealed` | The min-cut analysis event that detected the weak point |

---

## Storage Strategy

### On-Device

- Events are stored in a compact binary format to minimize storage overhead.
- The DAG is persisted incrementally (each append is immediately durable).
- Older sessions may be compressed (event payloads summarized) to save storage, but the hash chain and event metadata are always retained.

### Optional: Distributed Provenance

If `ruvnet/QuDAG` is integrated (Phase 5):
- The provenance DAG can be replicated across multiple devices.
- DAG consensus ensures tamper-evidence even if a single device is compromised.
- `ruvnet/Synaptic-Mesh` provides the P2P transport layer.

---

## Invariants

1. **Append-only.** Events are never modified or deleted. The DAG only grows.
2. **Hash integrity.** Every event's hash includes its payload and its predecessors' hashes. Any modification to a historical event invalidates all subsequent hashes.
3. **Complete coverage.** Every decision-making event in the system (ingestion, denial, prediction, adaptation, healing) is logged. There are no "dark" operations.
4. **Causal consistency.** An event's predecessors are always appended before the event itself. There are no forward references in the DAG.
5. **Latency budget.** Append operations complete in < 5 microseconds. Provenance logging never becomes a bottleneck for real-time operations.

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| ProvenanceLog | `synapse_graph::provenance` | (application layer) |
| DagStore | `synapse_graph::provenance::dag` | `ruvector-dag` v2.0.4 |
| ProvenanceQueryEngine | `synapse_graph::provenance::query` | `ruvector-dag` v2.0.4 |
| Hash chain computation | `synapse_graph::provenance::hash` | (std library SHA-256 or `ring`) |
| Distributed provenance (optional) | `synapse_graph::provenance::distributed` | `ruvnet/QuDAG`, `ruvnet/Synaptic-Mesh` |
