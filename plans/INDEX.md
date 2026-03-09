# SynapseGraph -- Architecture Documentation Index

## SPARC Specification

- [SPARC.md](./SPARC.md) -- Full Specification, Pseudocode, Architecture, Refinement, Completion

## Implementation Prompts

- [IMPLEMENTATION_PROMPTS.md](./IMPLEMENTATION_PROMPTS.md) -- 15 sequential prompts implementing all ADRs and DDDs

---

## Architecture Decision Records (ADRs)

| ADR | Title | Status | Key Decision |
|-----|-------|--------|--------------|
| [ADR-001](./adrs/ADR-001-three-tier-temporal-architecture.md) | Three-Tier Temporal Architecture | Accepted | Decompose learning into fast (<10ms), medium (min-hrs), slow (days-months) tiers |
| [ADR-002](./adrs/ADR-002-coherence-gating-hard-boundary.md) | Coherence Gating as Hard Boundary | Accepted | Mandatory gate before all storage; type-enforced via `GatedEmbedding` |
| [ADR-003](./adrs/ADR-003-proof-gated-graph-attention.md) | Proof-Gated Graph Attention | Accepted | Temporal + Manifold + Biological attention modules with verifiable proofs |
| [ADR-004](./adrs/ADR-004-two-tier-lora-adaptation.md) | Two-Tier LoRA Adaptation | Accepted | Rank-4 fast (<100us) + rank-16 slow (<10ms) with EWC++ |
| [ADR-005](./adrs/ADR-005-self-healing-graph-mincut.md) | Self-Healing Graph via Min-Cut | Accepted | Periodic min-cut analysis with reinforce/prune/rebalance operations |
| [ADR-006](./adrs/ADR-006-wasm-first-deployment.md) | WASM-First Deployment | Accepted | All modules compile to WASM; <4MB total; native fallback available |
| [ADR-007](./adrs/ADR-007-ruvector-ecosystem-dependency-selection.md) | ruvector Ecosystem Dependencies | Accepted | Unified dependency foundation from ruvector workspace (13 core crates) |

---

## Domain-Driven Design Documents (DDDs)

| DDD | Domain | Phase | Key Aggregates |
|-----|--------|-------|----------------|
| [DDD-001](./ddd/DDD-001-spike-ingestion-domain.md) | Spike Ingestion | Phase 1 | SpikeIngester, CoherenceGate, BCI Adapter (ACL) |
| [DDD-002](./ddd/DDD-002-vector-memory-domain.md) | Vector Memory | Phase 1 | VectorMemory, HnswStore, Collection |
| [DDD-003](./ddd/DDD-003-temporal-learning-domain.md) | Temporal Learning | Phase 2 | TemporalLearner, FastTier, MediumTier, SlowTier, TemporalRouter |
| [DDD-004](./ddd/DDD-004-cognitive-graph-domain.md) | Cognitive Graph | Phase 3 | CognitiveGraph, GnnEngine, GraphHealer, ShortcutRegistry |
| [DDD-005](./ddd/DDD-005-intent-prediction-domain.md) | Intent Prediction | Phase 4 | IntentPredictor, ShortcutMatcher, AttentionScorer |
| [DDD-006](./ddd/DDD-006-adaptation-domain.md) | Adaptation | Phase 4 | AdaptationEngine, FastLoraPath, SlowLoraPath |
| [DDD-007](./ddd/DDD-007-provenance-domain.md) | Provenance | Phase 5 | ProvenanceLog, DagStore, ProvenanceQueryEngine |

---

## Cross-Cutting Concerns

### Context Map Summary

```
Spike Ingestion (DDD-001)
    --> Vector Memory (DDD-002)        [GatedEmbedding]
    --> Temporal Learning (DDD-003)    [EmbeddingIngested event]
    --> Provenance (DDD-007)           [IngestEvent, DenyEvent]

Vector Memory (DDD-002)
    --> Cognitive Graph (DDD-004)      [EmbeddingCluster, KNN results]
    --> Intent Prediction (DDD-005)    [KNN query API]
    --> Provenance (DDD-007)           [EmbeddingStored event]

Temporal Learning (DDD-003)
    --> Cognitive Graph (DDD-004)      [TemporalSequence]
    --> Intent Prediction (DDD-005)    [TemporalContext API]
    --> Adaptation (DDD-006)           [DriftDetected event]
    --> Provenance (DDD-007)           [tier events]

Cognitive Graph (DDD-004)
    --> Intent Prediction (DDD-005)    [CognitiveShortcut registry]
    --> Provenance (DDD-007)           [discovery + healing events]

Intent Prediction (DDD-005)
    --> Adaptation (DDD-006)           [AdaptationSignal]
    --> Provenance (DDD-007)           [PredictionMade event]

Adaptation (DDD-006)
    --> Cognitive Graph (DDD-004)      [LoRA deltas to GNN, shared kernel]
    --> Intent Prediction (DDD-005)    [LoRA deltas to attention, shared kernel]
    --> Provenance (DDD-007)           [adaptation events]

Provenance (DDD-007)
    --> (terminal sink, no outbound)
```

### Crate Dependency Layers

```
Layer 0: ruvector-core, ruvector-math, ruvector-collections
Layer 1: ruvector-graph, ruvector-temporal-tensor, ruvector-dag
Layer 2: ruvector-gnn, ruvector-graph-transformer, ruvector-nervous-system,
         ruvector-attention, ruvector-attn-mincut, ruvector-mincut
Layer 3: ruvector-router-core, ruvector-sona, cognitum-gate-kernel, ruv-fann
Layer 4: ruvector-wasm, ruvector-gnn-wasm, ruvector-attention-wasm,
         ruvector-nervous-system-wasm, ruvector-learning-wasm,
         ruvector-tiny-dancer-core, ruvector-tiny-dancer-wasm
```
