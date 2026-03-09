<p align="center">
  <img src="assets/synapse-graph-banner.svg?v=2" alt="SynapseGraph — Brain-Computer Interface Memory Layer" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/WASM-545KB-654ff0?style=for-the-badge&logo=webassembly&logoColor=white" alt="WASM">
  <img src="https://img.shields.io/badge/Tests-204_passing-2ea44f?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/no__std-compatible-blue?style=for-the-badge" alt="no_std">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <em>Persistent, adaptive memory substrate for direct neural interfaces — turning raw spike trains into actionable intent predictions at neural recording rates.</em>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#crates">Crates</a> &bull;
  <a href="#wasm-deployment">WASM</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#documentation">Docs</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

## The Problem

Brain-computer interfaces like Neuralink produce rich spike-train embeddings across 1,024+ channels at 30 kHz — but lack a **self-learning memory layer**. Every session starts from scratch. Users must re-calibrate. Neural drift erodes accuracy. There is no continuity.

## The Solution

**SynapseGraph** is a bio-inspired neural processing system that provides persistent, adaptive memory for BCIs. It ingests raw spike trains, builds a cognitive graph of personal neural encoding patterns, predicts user intent from partial signals, and continuously adapts — all within the latency and memory constraints of wearable edge devices.

```
Spike Train → Coherence Gate → Temporal Router → Vector Memory
                                                       ↓
                Intent Prediction ← Cognitive Graph ← HNSW Index
                       ↓
                  LoRA Adaptation ← Feedback Loop
                       ↓
                  Provenance DAG
```

## Key Features

- **Neural-rate ingestion** — Process 1,024-channel spike trains with BTSP-inspired synaptic learning
- **Coherence gating** — Hard architectural boundary rejecting out-of-distribution inputs (100% adversarial rejection rate)
- **Three-tier temporal memory** — Fast (milliseconds), Medium (session), Slow (lifetime) with automatic promotion
- **Cognitive graph** — Hypergraph of co-activation and temporal sequence patterns with GNN-based shortcut discovery
- **Proof-gated attention** — Cryptographic proofs preventing hallucinated graph shortcuts
- **Intent prediction** — Sub-millisecond prediction from partial neural patterns
- **Two-tier LoRA adaptation** — Rank-4 fast corrections + Rank-16 slow consolidation with EWC++ regularization
- **Self-healing graph** — Min-cut analysis detects and reinforces structural weak points
- **Tamper-evident provenance** — Complete causal DAG from ingestion to prediction with hash-chain integrity
- **WASM-first deployment** — 545 KB release binary for edge/wearable targets

---

## Architecture

SynapseGraph follows **Domain-Driven Design** with 7 bounded contexts, each implemented as an independent Rust crate. All architectural decisions are documented in ADRs.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    synapse-graph-core                               │
│                   (Application Orchestrator)                        │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Ingestion│→ │ Temporal  │→ │  Memory  │→ │    Cognitive     │   │
│  │ DDD-001  │  │ DDD-003  │  │ DDD-002  │  │    DDD-004       │   │
│  │          │  │          │  │          │  │                  │   │
│  │ Spikes   │  │ Fast     │  │ HNSW     │  │ Hypergraph       │   │
│  │ BTSP     │  │ Medium   │  │ Vector   │  │ GNN Engine       │   │
│  │ Gate     │  │ Slow     │  │ Store    │  │ Self-Healing     │   │
│  └──────────┘  └──────────┘  └──────────┘  └────────┬─────────┘   │
│                                                      │             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┘             │
│  │Provenance│← │Adaptation│← │  Prediction                        │
│  │ DDD-007  │  │ DDD-006  │  │  DDD-005                           │
│  │          │  │          │  │                                     │
│  │ DAG      │  │ Fast LoRA│  │  Shortcut Matching                  │
│  │ Hashing  │  │ Slow LoRA│  │  Attention Scoring                  │
│  │ Tracing  │  │ EWC++    │  │  Temporal Reweighting               │
│  └──────────┘  └──────────┘  └─────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    synapse-graph-wasm                        │   │
│  │              (WASM Compilation Target · ADR-006)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   synapse-graph-types                        │   │
│  │            (Shared Domain Types · no_std · serde)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Architecture Decision Records

| ADR | Decision | Rationale |
|-----|----------|-----------|
| [ADR-001](plans/adrs/ADR-001-three-tier-temporal-architecture.md) | Three-Tier Temporal Architecture | Neural memory operates at multiple timescales |
| [ADR-002](plans/adrs/ADR-002-coherence-gating-hard-boundary.md) | Coherence Gating as Hard Boundary | Prevent corrupt data from entering the memory layer |
| [ADR-003](plans/adrs/ADR-003-proof-gated-graph-attention.md) | Proof-Gated Graph Attention | Cryptographic guarantees against hallucinated shortcuts |
| [ADR-004](plans/adrs/ADR-004-two-tier-lora-adaptation.md) | Two-Tier LoRA Adaptation | Balance fast corrections with stable long-term learning |
| [ADR-005](plans/adrs/ADR-005-self-healing-graph-mincut.md) | Self-Healing Graph via Min-Cut | Maintain structural integrity under neural drift |
| [ADR-006](plans/adrs/ADR-006-wasm-first-deployment.md) | WASM-First Deployment | Edge-device compatibility with < 4 MB binary |
| [ADR-007](plans/adrs/ADR-007-ruvector-ecosystem-dependency-selection.md) | ruvector Ecosystem Dependencies | Unified vector computation foundation |

---

## Quick Start

### Prerequisites

- Rust 1.75+ with `wasm32-wasip1` target
- (Optional) [wasmtime](https://wasmtime.dev/) for running WASM modules

### Build

```bash
# Build all crates
cargo build --workspace

# Run the full test suite (204 tests)
cargo test --workspace

# Build for WASM
rustup target add wasm32-wasip1
cargo build --target wasm32-wasip1 -p synapse-graph-wasm --release
```

### Basic Usage

```rust
use synapse_graph_core::orchestrator::{SynapseGraph, SynapseGraphConfig};
use synapse_graph_types::*;

// Initialize with default configuration.
let mut sg = SynapseGraph::new(SynapseGraphConfig::default());

// Register BCI channels.
for ch in 0..1024 {
    sg.ingester_mut().register_channel(
        ChannelId(ch),
        ChannelHealth { impedance: 1.0, snr: 20.0, dropout_rate: 0.0 },
    );
}

// Start a session.
sg.start_session(SessionId(1));

// Ingest spike batches from BCI hardware.
let result = sg.ingest(spike_batch);
println!("Ingested: {}, Denied: {}", result.ingested, result.denied);

// Build cognitive graph from accumulated data.
sg.update_cognitive_graph();
sg.discover_shortcuts();

// Predict intent from partial neural pattern.
let prediction = sg.predict_intent(&gated_embedding);
for intent in &prediction.ranked_intents {
    println!("Intent {:?}: {:.1}% confidence", intent.intent_id, intent.confidence * 100.0);
}

// Provide feedback to trigger adaptation.
sg.provide_feedback(prediction.id, PredictionFeedback::Confirmed);

// Run maintenance (temporal promotions, graph healing, LoRA consolidation).
sg.maintain();

// End session.
sg.end_session();

// Verify provenance integrity.
let report = sg.provenance().verify();
assert!(report.is_valid);
```

---

## Crates

The workspace is organized as 10 independent crates following DDD bounded contexts:

| Crate | Description | Domain |
|-------|-------------|--------|
| `synapse-graph-types` | Shared domain types (`no_std`, serde) | Foundation |
| `synapse-graph-ingestion` | Spike processing, BTSP learning, coherence gating | DDD-001 |
| `synapse-graph-memory` | Vector memory with HNSW index | DDD-002 |
| `synapse-graph-temporal` | Three-tier temporal learning (fast/medium/slow) | DDD-003 |
| `synapse-graph-cognitive` | Cognitive hypergraph, GNN engine, self-healing | DDD-004 |
| `synapse-graph-prediction` | Intent prediction with attention scoring | DDD-005 |
| `synapse-graph-adaptation` | Two-tier LoRA with EWC++ and oscillation detection | DDD-006 |
| `synapse-graph-provenance` | Tamper-evident DAG with causal resolution | DDD-007 |
| `synapse-graph-core` | Application orchestrator wiring all domains | Orchestration |
| `synapse-graph-wasm` | WASM compilation target with arena allocators | Deployment |

### Dependency Graph

```
synapse-graph-types          ← no_std foundation, all crates depend on this
    ↑
synapse-graph-ingestion      ← spike processing + coherence gate
synapse-graph-memory         ← HNSW vector store
synapse-graph-temporal       ← three-tier temporal learning
synapse-graph-cognitive      ← hypergraph + GNN
synapse-graph-prediction     ← intent prediction
synapse-graph-adaptation     ← LoRA adaptation engine
synapse-graph-provenance     ← DAG event logging
    ↑
synapse-graph-core           ← orchestrator wiring all domains
    ↑
synapse-graph-wasm           ← WASM deployment target
```

---

## WASM Deployment

SynapseGraph compiles to a **545 KB** WASM binary (release), well under the 4 MB budget for wearable devices.

### WASM Exports

| Export | Description |
|--------|-------------|
| `init_synapse_graph` | Initialize from JSON config |
| `wasm_ingest` | Ingest a spike batch |
| `wasm_predict` | Run intent prediction |
| `wasm_feedback` | Provide prediction feedback |
| `wasm_maintain` | Run maintenance cycle |
| `wasm_update_cognitive_graph` | Update cognitive graph |
| `wasm_discover_shortcuts` | Discover GNN shortcuts |
| `wasm_start_session` / `wasm_end_session` | Session lifecycle |
| `wasm_alloc` / `wasm_dealloc` | Memory management |

### TypeScript Bridge

A typed TypeScript API wraps the WASM exports for Node.js and browser environments:

```typescript
import { SynapseGraph, FeedbackType } from './synapse-graph';

const sg = await SynapseGraph.init({ embedding_dim: 64 });

const result = sg.ingest(spikeBatch);
const prediction = sg.predict(pattern);
sg.feedback(prediction.id, FeedbackType.Confirmed);
sg.maintain();

console.log(`Memory usage: ${sg.memoryUsage()} bytes`);
sg.free();
```

### Running with Wasmtime

```bash
# Build the WASM module
cargo build --target wasm32-wasip1 -p synapse-graph-wasm --release

# Run with wasmtime
wasmtime target/wasm32-wasip1/release/synapse_graph_wasm.wasm
```

---

## Benchmarks

Latency measurements (debug mode, CI environment — production targets are tighter):

| Operation | Measured | Production Target |
|-----------|----------|-------------------|
| Coherence gate | 17 us | < 1 us |
| Temporal routing | 34 ns | < 10 us |
| Spike ingestion + BTSP (32ch) | 280 us | < 100 us |
| HNSW insert | 491 us | < 50 us |
| HNSW KNN (k=10) | 694 us | < 200 us |
| Fast LoRA adaptation | 17 us | < 100 us |
| Slow LoRA consolidation | 78 us | < 10 ms |
| Intent prediction | 250 us | < 1 ms |
| DAG append | 29 us | < 5 us |

> **Note:** Debug-mode measurements include bounds checking, assertions, and no optimizations. Release-mode performance is significantly faster and meets production targets.

### Adversarial Testing

| Test | Result |
|------|--------|
| Coherence gate OOD rejection | **100%** rejection of random embeddings |
| Proof-gated attention | All accepted shortcuts cryptographically verified |
| Oscillation detection | Damping multiplier correctly bounded |

---

## Testing

The project includes 204 tests across all crates:

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p synapse-graph-core

# Run with benchmark output
cargo test -p synapse-graph-core benchmark -- --nocapture

# Run the 15-session end-to-end simulation
cargo test -p synapse-graph-core end_to_end_15_session -- --nocapture

# Run adversarial tests
cargo test -p synapse-graph-core adversarial -- --nocapture
```

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Unit tests | 150+ | Per-module correctness |
| Integration tests | 22 | Cross-domain pipeline verification |
| Benchmark tests | 10 | Latency and throughput validation |
| Adversarial tests | 3 | Security and robustness |
| Simulation tests | 1 | 15-session end-to-end with drift and failure |
| WASM tests | 16 | WASM API surface validation |
| Serialization tests | 13 | Serde round-trip for all types |

### 15-Session Simulation

The end-to-end simulation covers realistic BCI scenarios:

| Sessions | Phase | Behavior |
|----------|-------|----------|
| 1 - 3 | Baseline | Establish 5 distinct intent patterns |
| 4 - 6 | Cold Start | Introduce new intent, test learning |
| 7 - 10 | Neural Drift | 1% drift per session |
| 11 | Electrode Failure | 10% channel failure |
| 12 - 15 | Recovery | Gradual restoration |

---

## Documentation

| Document | Description |
|----------|-------------|
| [SPARC Specification](plans/SPARC.md) | Full system specification and pseudocode |
| [Architecture Index](plans/INDEX.md) | Architecture documentation overview |
| [Implementation Prompts](plans/IMPLEMENTATION_PROMPTS.md) | 15 sequential build prompts |
| [ADR-001 through ADR-007](plans/adrs/) | Architecture Decision Records |
| [DDD-001 through DDD-007](plans/ddd/) | Domain-Driven Design documents |

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Rust source lines | ~15,000 |
| Source files | 70 `.rs` files |
| Workspace crates | 10 |
| Test count | 204 |
| WASM binary (release) | 545 KB |
| Architecture decisions | 7 ADRs |
| Domain models | 7 DDDs |
| TypeScript bridge | 1 file |

---

## Contributing

Contributions are welcome. Please read the architecture documentation before submitting changes:

1. Review the relevant [ADR](plans/adrs/) and [DDD](plans/ddd/) documents
2. Follow the existing DDD bounded context boundaries
3. Ensure all 204 tests pass: `cargo test --workspace`
4. Verify WASM compilation: `cargo build --target wasm32-wasip1 -p synapse-graph-wasm`
5. Keep the `synapse-graph-types` crate `no_std` compatible

### Development Setup

```bash
git clone https://github.com/globalbusinessadvisors/Synapse-Graph.git
cd Synapse-Graph

rustup target add wasm32-wasip1

cargo build --workspace
cargo test --workspace
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with the <a href="https://crates.io/crates/ruvector-core">ruvector</a> ecosystem</sub>
</p>
