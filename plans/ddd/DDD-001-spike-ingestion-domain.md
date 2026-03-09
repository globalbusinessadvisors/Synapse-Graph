# DDD-001: Spike Ingestion Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R1, R9 | ADR-002
**Implementing Phase:** Phase 1

---

## Domain Overview

The Spike Ingestion domain is the entry point of the SynapseGraph system. It receives raw spike-train data from BCI hardware, preprocesses it through bio-inspired spiking neural networks, validates it through the coherence gate, and produces clean, typed embeddings ready for downstream storage and analysis.

This domain owns the boundary between the physical (BCI hardware) and digital (SynapseGraph memory layer) worlds.

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Spike Train** | A time-series of neural action potentials recorded from one or more electrodes. Represented as a sequence of (timestamp, channel_id, amplitude) tuples. |
| **Spike Batch** | A collection of spike trains from multiple channels covering a single time window (typically 0.33 ms at 30 kHz). |
| **Embedding** | A high-dimensional float vector (`Vec<f32>`, dimension 256-1024) derived from a spike batch by the nervous system processor. |
| **Channel** | A single recording electrode with a unique ID, spatial position, and impedance characteristics. |
| **BTSP** | Behavioral Time-Scale Plasticity. A learning rule applied during spike processing that strengthens connections between temporally correlated spike patterns. |
| **Coherence Verdict** | A binary PERMIT or DENY decision on an embedding, with an associated reason code. |
| **Gated Embedding** | An embedding that has passed the coherence gate. This is a distinct type that can only be constructed by the gate -- downstream modules cannot accept raw embeddings. |
| **Channel Health** | A per-channel metric tracking impedance, signal-to-noise ratio, and dropout frequency. |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 SPIKE INGESTION CONTEXT                           |
|                                                                   |
|  +-----------------+                                              |
|  | BCI Adapter     |  Anti-Corruption Layer                      |
|  | (port/adapter)  |  Translates vendor-specific BCI formats     |
|  +--------+--------+  into domain SpikeBatch                     |
|           |                                                       |
|           v                                                       |
|  +------------------+                                             |
|  | SpikeIngester    |  Domain Service                             |
|  | (aggregate root) |  Orchestrates preprocessing + gating        |
|  +--------+---------+                                             |
|           |                                                       |
|     +-----+------+                                                |
|     |            |                                                |
|     v            v                                                |
|  +----------+ +----------------+                                  |
|  | Nervous  | | CoherenceGate  |                                  |
|  | System   | | (value object  |                                  |
|  | Processor| |  factory)      |                                  |
|  +----------+ +-------+--------+                                  |
|                       |                                           |
|              PERMIT   |   DENY                                    |
|                |      |     |                                     |
|                v      |     v                                     |
|        GatedEmbedding |  DenyEvent --> [Provenance Context]       |
|                       |                                           |
+------------------------------------------------------------------+
        |
        v  (Published Domain Event: EmbeddingIngested)
  [Temporal Learning Context]
  [Vector Memory Context]
```

### Context Map

| Relationship | Upstream | Downstream | Type |
|-------------|----------|------------|------|
| Spike Ingestion -> Vector Memory | Spike Ingestion | Vector Memory (DDD-002) | Published Language (GatedEmbedding) |
| Spike Ingestion -> Temporal Learning | Spike Ingestion | Temporal Learning (DDD-003) | Published Language (EmbeddingIngested event) |
| Spike Ingestion -> Provenance | Spike Ingestion | Provenance (DDD-006) | Published Language (IngestEvent, DenyEvent) |
| BCI Hardware -> Spike Ingestion | External (BCI) | Spike Ingestion | Anti-Corruption Layer (BCI Adapter) |

---

## Domain Model

### Aggregates

#### SpikeIngester (Aggregate Root)

```rust
/// Aggregate root for the spike ingestion pipeline.
/// Owns the nervous system processor and coherence gate.
/// Invariant: every embedding that exits this aggregate has passed the coherence gate.
struct SpikeIngester {
    nervous_system: NervousSystemProcessor,  // ruvector-nervous-system
    coherence_gate: CoherenceGate,           // cognitum-gate-kernel
    channel_registry: ChannelRegistry,
    config: IngestionConfig,
}

impl SpikeIngester {
    /// Process a batch of spike trains.
    /// Returns: Vec<GatedEmbedding> (passed) + Vec<DenyEvent> (rejected)
    fn ingest(&mut self, batch: SpikeBatch) -> IngestionResult;
}
```

### Entities

#### Channel

```rust
/// A recording electrode with tracked health state.
/// Identity: channel_id (stable across sessions).
struct Channel {
    id: ChannelId,
    position: SpatialPosition,  // (x, y, z) in electrode array
    health: ChannelHealth,      // impedance, SNR, dropout rate
    last_active: Timestamp,
}
```

### Value Objects

#### SpikeBatch

```rust
/// Immutable collection of spike trains for a single time window.
struct SpikeBatch {
    timestamp: Timestamp,
    session_id: SessionId,
    spikes: Vec<SpikeEvent>,  // (channel_id, amplitude, sub_timestamp)
    duration_us: u32,         // batch duration in microseconds
}
```

#### GatedEmbedding

```rust
/// An embedding that has been verified by the coherence gate.
/// Cannot be constructed outside the coherence gate module.
/// This type is the ONLY input accepted by downstream contexts.
struct GatedEmbedding {
    vector: Vec<f32>,          // dimension d (256-1024)
    channel_id: ChannelId,
    timestamp: Timestamp,
    session_id: SessionId,
    coherence_score: f32,      // 0.0-1.0, how confidently the gate approved
    // private constructor -- only CoherenceGate can create this
}
```

#### CoherenceVerdict

```rust
/// Result of coherence gate evaluation.
enum CoherenceVerdict {
    Permit { score: f32 },
    Deny { reason: DenyReason },
}

enum DenyReason {
    ManifoldOutlier { distance: f32, threshold: f32 },
    TemporalDiscontinuity { delta: f32, expected_max: f32 },
    ChannelUnhealthy { channel_id: ChannelId, health: ChannelHealth },
    SaturatedSignal,
    ZeroSignal,
}
```

### Domain Events

```rust
/// Published when an embedding passes the coherence gate and is ready for storage.
struct EmbeddingIngested {
    embedding_id: EmbeddingId,
    gated_embedding: GatedEmbedding,
    btsp_metadata: BtspMetadata,  // learning rule outputs from nervous system
}

/// Published when an embedding is rejected by the coherence gate.
struct EmbeddingDenied {
    embedding_id: EmbeddingId,
    reason: DenyReason,
    channel_id: ChannelId,
    timestamp: Timestamp,
}

/// Published when a channel's health state changes.
struct ChannelHealthChanged {
    channel_id: ChannelId,
    previous: ChannelHealth,
    current: ChannelHealth,
}
```

---

## Domain Services

### NervousSystemProcessor

**Backed by:** `ruvector-nervous-system` v2.0.4 / `ruvector-nervous-system-wasm` v2.0.4

Transforms raw spike trains into embeddings using bio-inspired spiking neural networks with BTSP learning enabled. This is a stateful service -- BTSP learning modifies internal synaptic weights with each batch.

```
SpikeBatch --> NervousSystemProcessor --> ProcessedBatch {
    embeddings: Vec<RawEmbedding>,
    btsp_metadata: BtspMetadata,
    timing: TimingMetadata,
}
```

### CoherenceGate

**Backed by:** `cognitum-gate-kernel` v0.1.1, `ruvector-math` v2.0.4

Evaluates each raw embedding against manifold consistency, temporal continuity, and channel health criteria. Produces GatedEmbedding (PERMIT) or DenyEvent (DENY).

**Performance contract:** < 1 microsecond per evaluation.

---

## Anti-Corruption Layer

### BCI Adapter

The BCI Adapter translates vendor-specific data formats into the domain's `SpikeBatch` representation. Each BCI vendor (Neuralink, Blackrock, etc.) gets a separate adapter implementation.

```rust
trait BciAdapter {
    fn translate(&self, raw_data: &[u8]) -> Result<SpikeBatch, AdapterError>;
    fn channel_map(&self) -> &ChannelMap;
    fn sampling_rate_hz(&self) -> u32;
}
```

This is a classic ACL pattern: the domain never sees vendor-specific types.

---

## Invariants

1. **No raw embedding escapes the ingestion context.** All downstream contexts receive `GatedEmbedding`, which can only be constructed by the `CoherenceGate`.
2. **Every denied embedding is logged.** `EmbeddingDenied` events are published for every DENY verdict.
3. **Channel health is monotonically tracked.** Health changes produce `ChannelHealthChanged` events. A channel marked unhealthy remains so until explicitly reset (e.g., after recalibration).
4. **BTSP learning is always active.** The nervous system processor applies BTSP on every batch; there is no "inference-only" mode during ingestion.

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| SpikeIngester | `synapse_graph::ingestion` | (application layer) |
| NervousSystemProcessor | `synapse_graph::ingestion::nervous_system` | `ruvector-nervous-system` v2.0.4 |
| NervousSystemProcessor (WASM) | `synapse_graph::ingestion::nervous_system_wasm` | `ruvector-nervous-system-wasm` v2.0.4 |
| CoherenceGate | `synapse_graph::ingestion::gate` | `cognitum-gate-kernel` v0.1.1 |
| Manifold consistency check | `synapse_graph::ingestion::gate::manifold` | `ruvector-math` v2.0.4 |
| BCI Adapter | `synapse_graph::ingestion::adapters` | (application layer, per vendor) |
