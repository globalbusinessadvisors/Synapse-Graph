# ADR-006: WASM-First Deployment Strategy

**Status:** Accepted
**Date:** 2026-03-09
**Deciders:** SynapseGraph Architecture Team
**SPARC Reference:** Specification R8

---

## Context

SynapseGraph must run on wearable edge devices co-located with BCI hardware. These devices are:

- **Resource-constrained:** ARM Cortex-M or RISC-V class processors, 64 MB RAM, battery-powered.
- **Heterogeneous:** Different BCI vendors will use different hardware platforms.
- **Updatable:** The model and logic must be updatable without physical access to the device.
- **Safety-critical:** Incorrect behavior could cause incorrect neural interface responses.

Native compilation for each target platform is possible but creates a combinatorial explosion of build targets, testing matrices, and deployment pipelines.

## Decision

We adopt a WASM-first deployment strategy: the entire inference and learning pipeline compiles to WebAssembly and runs on a WASM runtime (e.g., wasmtime, wasmer, or a bare-metal WASM interpreter).

### WASM Module Architecture

Each major subsystem is compiled as a separate WASM module:

| Module | Crate | WASM Crate | Build Size Target |
|--------|-------|------------|-------------------|
| Core vector operations | `ruvector-core` v2.0.5 | `ruvector-wasm` v0.1.29 | < 500 KB |
| GNN inference | `ruvector-gnn` v2.0.5 | `ruvector-gnn-wasm` v2.0.4 | < 800 KB |
| Attention mechanisms | `ruvector-attention` v2.0.4 | `ruvector-attention-wasm` v2.0.4 | < 400 KB |
| Spiking neural system | `ruvector-nervous-system` v2.0.4 | `ruvector-nervous-system-wasm` v2.0.4 | < 600 KB |
| On-device learning | `ruvector-sona` v0.1.6 | `ruvector-learning-wasm` v2.0.4 | < 400 KB |
| Neural routing | `ruvector-tiny-dancer-core` v2.0.4 | `ruvector-tiny-dancer-wasm` v2.0.4 | < 300 KB |
| Coherence gate | `cognitum-gate-kernel` v0.1.1 | (self-contained `no_std`) | < 50 KB |
| **Total** | | | **< 4 MB** |

### WASM Runtime Requirements

- **WASI support:** For filesystem access (persistent storage of embeddings and LoRA weights).
- **SIMD:** WASM SIMD128 for vector operations (supported by wasmtime, wasmer).
- **Threads:** Optional; single-threaded operation is the baseline. Multi-threaded WASM (shared memory + atomics) for devices that support it.
- **Memory:** 64 MB linear memory limit.

### Native Fallback

For development, testing, and high-performance deployments (e.g., desktop workstations processing BCI data from a tethered device), the same Rust crates compile natively without WASM. The WASM crates are thin binding layers; the core logic is target-independent.

`ruv-fann` v0.2.0 provides the CPU-native neural engine for non-WASM targets.
`ruvector-sparse-inference` v0.1.31 provides PowerInfer-style sparse execution for both WASM and native.

### JavaScript Bridge

For browser-based visualization and configuration:

- `ruvector` v0.2.11 (npm): Node.js entry point with native/WASM fallback
- `@ruvector/gnn` v0.1.25 (npm): GNN bindings via NAPI-RS
- `@ruvector/rvf-wasm` v0.1.6 (npm): WASM microkernel for browser vector operations
- `@ruvector/rvf` v0.2.0 (npm): Unified TypeScript SDK

### Update Strategy

WASM modules can be updated independently via over-the-air (OTA) updates:
1. New WASM binary is signed and transmitted to the device.
2. The WASM runtime loads the new module alongside the old one.
3. A canary test runs the new module against recent embeddings.
4. If the canary passes, the old module is replaced. If not, the update is rolled back.
5. All update events are logged to the DAG provenance log.

## Consequences

### Positive

- Single build target (wasm32-wasi) covers all hardware platforms.
- WASM sandbox provides memory safety guarantees beyond Rust's own safety.
- Independent module updates reduce deployment risk.
- Browser-based tooling can reuse the same WASM modules.

### Negative

- WASM execution is 1.5-3x slower than native for compute-heavy operations. This is why the performance budget in ADR-001 is designed with headroom.
- WASM SIMD support varies by runtime. Fall back to scalar operations on limited runtimes.
- Debugging WASM on embedded devices is harder than native debugging.

### Risks

- **WASM runtime availability:** Not all embedded platforms have a mature WASM runtime. Mitigation: the core crates compile natively; WASM is the preferred but not the only target.
- **Linear memory fragmentation:** Long-running WASM modules may fragment their 64 MB linear memory. Mitigation: arena allocators for short-lived data (fast tier ring buffer); explicit deallocation for graph nodes.

## Alternatives Considered

1. **Native cross-compilation per target:** Maximum performance but creates N build targets x M test configurations. Not scalable across BCI vendor ecosystem.
2. **eBPF:** Interesting for kernel-level performance but not portable across the embedded platforms we target.
3. **Lua / MicroPython:** Too slow for < 100 us latency requirements. No type safety.
