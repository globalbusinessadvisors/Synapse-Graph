# DDD-002: Vector Memory Domain

**Status:** Accepted
**Date:** 2026-03-09
**SPARC Reference:** Specification R2 | ADR-006, ADR-007
**Implementing Phase:** Phase 1

---

## Domain Overview

The Vector Memory domain is the persistent storage substrate of SynapseGraph. It manages the HNSW-indexed vector database that stores all neural embeddings, organized into collections with temporal metadata. This domain is the system of record for neural data -- all other domains query it for embeddings.

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| **Vector Store** | The HNSW-indexed persistent database storing all gated embeddings. |
| **Collection** | A named, configurable partition of the vector store (e.g., per-user, per-session, per-brain-region). |
| **HNSW Index** | Hierarchical Navigable Small World graph providing approximate nearest-neighbor search in O(log N) time. |
| **Temporal Metadata** | Per-embedding metadata including temporal tier, timestamp, channel, session, and coherence score. |
| **KNN Query** | K-nearest-neighbors search returning the k most similar embeddings to a query vector. |
| **Neighborhood** | The HNSW graph neighbors of an embedding -- used for clustering and cognitive graph construction. |
| **Embedding Cluster** | A group of embeddings that are neighbors in the HNSW graph and share temporal proximity. |

---

## Bounded Context

```
+------------------------------------------------------------------+
|                 VECTOR MEMORY CONTEXT                              |
|                                                                   |
|  Inbound:                                                         |
|    GatedEmbedding (from Spike Ingestion)                          |
|    KNN Query (from Intent Prediction, Cognitive Graph)            |
|                                                                   |
|  +---------------------+                                         |
|  | VectorMemory        |  Aggregate Root                         |
|  | (owns store +       |  Single point of entry for all          |
|  |  collections)       |  embedding storage and retrieval        |
|  +----------+----------+                                         |
|             |                                                     |
|    +--------+--------+                                            |
|    |                 |                                            |
|    v                 v                                            |
|  +----------+  +----------------+                                 |
|  | HnswStore|  | CollectionMgr  |                                 |
|  | (entity) |  | (entity)       |                                 |
|  +----------+  +----------------+                                 |
|                                                                   |
|  Outbound:                                                        |
|    EmbeddingStored event --> [Provenance Context]                  |
|    ClusterResult --> [Cognitive Graph Context]                     |
|                                                                   |
+------------------------------------------------------------------+
```

### Context Map

| Relationship | Upstream | Downstream | Type |
|-------------|----------|------------|------|
| Spike Ingestion -> Vector Memory | Spike Ingestion (DDD-001) | Vector Memory | Conformist (accepts GatedEmbedding as-is) |
| Vector Memory -> Cognitive Graph | Vector Memory | Cognitive Graph (DDD-003) | Published Language (ClusterResult, KNN results) |
| Vector Memory -> Intent Prediction | Vector Memory | Intent Prediction (DDD-004) | Open Host Service (KNN query API) |
| Vector Memory -> Provenance | Vector Memory | Provenance (DDD-006) | Published Language (EmbeddingStored event) |

---

## Domain Model

### Aggregates

#### VectorMemory (Aggregate Root)

```rust
/// Aggregate root for persistent vector storage.
/// Owns the HNSW index and collection manager.
/// Invariant: only GatedEmbeddings can be inserted.
struct VectorMemory {
    store: HnswStore,                    // ruvector-core
    collections: CollectionManager,       // ruvector-collections
    config: VectorMemoryConfig,
}

impl VectorMemory {
    /// Insert a gated embedding with temporal metadata.
    fn insert(&mut self, embedding: GatedEmbedding, meta: TemporalMeta)
        -> Result<EmbeddingId, InsertError>;

    /// KNN search returning k nearest neighbors with metadata.
    fn search_knn(&self, query: &[f32], k: usize)
        -> Vec<SearchResult>;

    /// Find embeddings temporally adjacent to the given cluster.
    fn find_temporally_adjacent(&self, cluster: &EmbeddingCluster, window: Duration)
        -> Vec<TemporalNeighbor>;

    /// Cluster recent embeddings using HNSW neighborhood structure.
    fn cluster_recent(&self, threshold: f32)
        -> Vec<EmbeddingCluster>;
}
```

### Entities

#### HnswStore

```rust
/// The HNSW-indexed vector database.
/// Identity: singleton per VectorMemory instance.
struct HnswStore {
    index: ruvector_core::HnswIndex,
    dimension: usize,          // embedding dimension (256-1024)
    ef_construction: usize,    // HNSW build parameter
    ef_search: usize,          // HNSW query parameter
    m: usize,                  // HNSW max connections per layer
    total_embeddings: u64,
}
```

#### Collection

```rust
/// A named partition of the vector store.
/// Identity: collection_id.
struct Collection {
    id: CollectionId,
    name: String,
    filter: CollectionFilter,   // e.g., session_id, channel range, temporal tier
    embedding_count: u64,
    created_at: Timestamp,
}
```

### Value Objects

#### TemporalMeta

```rust
/// Metadata attached to every stored embedding.
struct TemporalMeta {
    tier: TemporalTier,        // Fast, Medium, Slow
    timestamp: Timestamp,
    channel_id: ChannelId,
    session_id: SessionId,
    coherence_score: f32,
}
```

#### SearchResult

```rust
/// Result of a KNN query.
struct SearchResult {
    embedding_id: EmbeddingId,
    vector: Vec<f32>,
    distance: f32,             // L2 or cosine distance
    metadata: TemporalMeta,
}
```

#### EmbeddingCluster

```rust
/// A group of related embeddings identified by HNSW neighborhood analysis.
struct EmbeddingCluster {
    centroid: Vec<f32>,
    member_ids: Vec<EmbeddingId>,
    metadata: ClusterMetadata,
    temporal_span: (Timestamp, Timestamp),
}
```

### Domain Events

```rust
/// Published when an embedding is successfully stored.
struct EmbeddingStored {
    embedding_id: EmbeddingId,
    collection_id: CollectionId,
    metadata: TemporalMeta,
    hnsw_neighbors: Vec<EmbeddingId>,  // immediate HNSW graph neighbors
}

/// Published when memory pressure triggers eviction.
struct EmbeddingsEvicted {
    collection_id: CollectionId,
    count: u64,
    reason: EvictionReason,   // MemoryPressure, TierExpiry, ManualPrune
}
```

---

## Domain Services

### Persistence Strategy

The HNSW index and metadata are persisted to device storage via `ruvector-core`'s built-in persistence. On device restart:

1. HNSW index is memory-mapped from storage.
2. Metadata is loaded into an in-memory index.
3. Collections are reconstructed from metadata.

### Memory Budget Management

Under the 64 MB device constraint, the vector store dynamically manages its memory allocation:

- **Fast-tier embeddings:** Stored with full f32 precision. Oldest evicted when fast-tier budget is exhausted.
- **Medium-tier embeddings:** Stored with 8-bit quantization via `ruvector-temporal-tensor`.
- **Slow-tier embeddings:** Stored with 4-bit quantization.
- **HNSW parameters** (`ef_construction`, `M`) are reduced under memory pressure to decrease index memory overhead.

---

## Invariants

1. **Type-safe insertion.** Only `GatedEmbedding` values can be inserted. Raw embeddings are rejected at compile time.
2. **Metadata completeness.** Every stored embedding has complete `TemporalMeta` -- no null fields.
3. **Collection isolation.** Collections are logical partitions; a KNN search can span collections but filters are always applied.
4. **Persistence guarantee.** An `EmbeddingStored` event is only published after the embedding is durably written to storage.

---

## Crate-to-Module Mapping

| Domain Concept | Rust Module | Backing Crate |
|---------------|-------------|---------------|
| VectorMemory | `synapse_graph::memory` | (application layer) |
| HnswStore | `synapse_graph::memory::hnsw` | `ruvector-core` v2.0.5 |
| HnswStore (WASM) | `synapse_graph::memory::hnsw_wasm` | `ruvector-wasm` v0.1.29 |
| CollectionManager | `synapse_graph::memory::collections` | `ruvector-collections` v2.0.4 |
| Quantization | `synapse_graph::memory::quantization` | `ruvector-temporal-tensor` v2.0.4 |
