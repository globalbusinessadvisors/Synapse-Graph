# ADR-007: ruvector Ecosystem as Unified Dependency Foundation

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** All Specification Requirements

---

## Context

SynapseGraph requires capabilities spanning vector databases, graph neural networks, temporal learning, attention mechanisms, WASM compilation, spiking neural networks, and provenance logging. These capabilities could be sourced from:

1. Independent open-source libraries (each with its own API surface, versioning, and maintenance).
2. A unified ecosystem where crates are designed to interoperate.

The ruvector ecosystem (`ruvnet/ruvector` workspace) provides all required capabilities as workspace members with aligned versions, shared types, and tested interoperability.

## Decision

We adopt the ruvector ecosystem as the unified dependency foundation for SynapseGraph. All core functionality is sourced from ruvector crates, with `ruvnet/ruv-FANN` providing the neural network runtime.

### Required GitHub Repositories

| Repository | Role | Necessity |
|------------|------|-----------|
| `ruvnet/ruvector` | Core workspace containing all crates below | **Mandatory** |
| `ruvnet/ruv-FANN` | Neural network engine (LSTM, TCN, Transformers), WASM neural runtime, `ruv-swarm` orchestration | **Mandatory** |
| `ruvnet/Synaptic-Mesh` | Distributed P2P DAG for multi-device coordination | Optional (Phase 5) |
| `ruvnet/QuDAG` | Tamper-evident DAG consensus for spike-train provenance | Optional (Phase 5) |
| `ruvnet/dspy.ts` | JS/TS declarative DSL for cognitive pattern configuration | Optional (JS layer) |

### Required Rust Crates (Minimal Set)

**Layer 0 -- Foundation:**

| Crate | Version | Role |
|-------|---------|------|
| `ruvector-core` | 2.0.5 | HNSW vector database; primary persistent store |
| `ruvector-math` | 2.0.4 | Optimal transport, information geometry, manifold metrics |
| `ruvector-collections` | 2.0.4 | Collection management for the memory layer |

**Layer 1 -- Graph & Temporal:**

| Crate | Version | Role |
|-------|---------|------|
| `ruvector-graph` | 2.0.4 | Distributed hypergraph database |
| `ruvector-temporal-tensor` | 2.0.4 | Tiered temporal compression |
| `ruvector-dag` | 2.0.4 | DAG provenance logging |

**Layer 2 -- Neural Processing:**

| Crate | Version | Role |
|-------|---------|------|
| `ruvector-gnn` | 2.0.5 | GNN on HNSW topology |
| `ruvector-graph-transformer` | 2.0.4 | Proof-gated graph attention (8 modules) |
| `ruvector-nervous-system` | 2.0.4 | Bio-inspired spiking NN, BTSP, EWC |
| `ruvector-attention` | 2.0.4 | Geometric, graph, sparse attention |
| `ruvector-attn-mincut` | 2.0.4 | Min-cut gated attention |
| `ruvector-mincut` | 2.0.4 | Subpolynomial dynamic min-cut |

**Layer 3 -- Routing & Adaptation:**

| Crate | Version | Role |
|-------|---------|------|
| `ruvector-router-core` | 2.0.4 | Neural routing engine |
| `ruvector-sona` | 0.1.6 | Two-tier LoRA + EWC++ |
| `cognitum-gate-kernel` | 0.1.1 | No-std WASM coherence gate |
| `ruv-fann` | 0.2.0 | CPU-native neural engine |

**Layer 4 -- WASM Bindings:**

| Crate | Version | Role |
|-------|---------|------|
| `ruvector-wasm` | 0.1.29 | Core WASM bindings |
| `ruvector-gnn-wasm` | 2.0.4 | GNN WASM bindings |
| `ruvector-attention-wasm` | 2.0.4 | Attention WASM bindings |
| `ruvector-nervous-system-wasm` | 2.0.4 | Spiking NN WASM bindings |
| `ruvector-learning-wasm` | 2.0.4 | On-device learning WASM bindings |
| `ruvector-tiny-dancer-core` | 2.0.4 | FastGRNN neural routing |
| `ruvector-tiny-dancer-wasm` | 2.0.4 | FastGRNN WASM bindings |

**Extended Set (Phase 5+):**

| Crate | Version | Role |
|-------|---------|------|
| `ruvector-sparse-inference` | 0.1.31 | PowerInfer-style sparse inference |
| `ruvllm` | 2.0.4 | Local LLM inference (paged attention, KV cache) |

### Required npm Packages (Minimal Set)

| Package | Version | Role |
|---------|---------|------|
| `ruvector` | 0.2.11 | Node.js vector DB entry point |
| `@ruvector/gnn` | 0.1.25 | GNN Node.js bindings (NAPI-RS) |
| `@ruvector/rvf-wasm` | 0.1.6 | WASM microkernel for edge/browser |

### Extended npm Packages (Optional)

| Package | Version | Role |
|---------|---------|------|
| `@ruvector/graph-node` | 2.0.2 | Graph DB Node.js bindings (Cypher) |
| `@ruvector/rvf` | 0.2.0 | Unified TypeScript SDK |
| `@ruvector/gnn-wasm` | 0.1.0 | WASM GNN for browser |
| `ruvector-graph-transformer-wasm` | 2.0.4 | WASM proof-gated graph attention |
| `@ruvnet/ruvector-verified-wasm` | 0.1.1 | Proof-carrying vector ops |
| `ruvector-extensions` | 0.1.0 | Temporal tracking additions |
| `midstreamer` | 0.2.4 | WASM temporal analysis (DTW, LCS) |
| `agentic-flow` | 2.0.7 | Multi-agent orchestration |
| `@claude-flow/plugin-neural-coordination` | 3.0.0-alpha.1 | Neural coordination plugin |
| `@claude-flow/plugin-cognitive-kernel` | 3.0.0-alpha.1 | Cognitive kernel plugin |

### Version Alignment

All v2.0.x crates are from the same ruvector workspace release. The following crates have independent versioning due to being published earlier:
- `ruvector-sona` v0.1.6 -- pre-2.0 but API-compatible
- `ruvector-wasm` v0.1.29 -- independently versioned WASM layer
- `ruvector-sparse-inference` v0.1.31 -- independently versioned
- `cognitum-gate-kernel` v0.1.1 -- standalone `no_std` crate

## Consequences

### Positive

- Unified type system across all crates eliminates serialization overhead at module boundaries.
- Workspace-level CI ensures cross-crate compatibility.
- Single upstream (`ruvnet/ruvector`) for bug reports and feature requests.
- Consistent API patterns reduce cognitive load during development.

### Negative

- Strong coupling to a single ecosystem. If ruvector development stalls, SynapseGraph is affected.
- Version pinning is critical; a breaking change in `ruvector-core` cascades to all dependent crates.
- Some crates may include functionality we don't need, adding to binary size.

### Risks

- **Upstream breaking changes:** Mitigation: pin exact versions in `Cargo.toml`; maintain a fork of `ruvnet/ruvector` as a fallback.
- **License changes:** Mitigation: verify license compatibility before each version bump.
- **WASM build size:** Mitigation: use `wasm-opt` and LTO; the modular WASM architecture (ADR-006) allows loading only needed modules.

## Alternatives Considered

1. **Mix-and-match from independent ecosystems:** E.g., `qdrant` for vectors, `petgraph` for graphs, `burn` for neural networks. This creates impedance mismatches at every boundary, requires custom serialization, and loses the GNN-on-HNSW integration that `ruvector-gnn` provides natively.
2. **Build from scratch:** Maximum control but unrealistic development timeline for a system of this complexity.
3. **Use ruvector for core, independent crates for periphery:** Possible but the tight integration between GNN, attention, and temporal learning in the ruvector ecosystem is a significant advantage.
