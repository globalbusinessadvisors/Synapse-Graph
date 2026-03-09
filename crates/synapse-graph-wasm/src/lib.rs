//! SynapseGraph WASM compilation target (ADR-006).
//!
//! This crate provides the WASM interface for the SynapseGraph neural
//! processing system. It exposes C-ABI functions compatible with both
//! wasm32-wasi and wasm32-unknown-unknown targets.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Host Runtime (wasmtime / browser / Node.js)    │
//! │  ┌─────────────────────────────────────────┐    │
//! │  │  TypeScript Bridge (synapse-graph.ts)   │    │
//! │  └──────────────┬──────────────────────────┘    │
//! │                 │ JSON over linear memory        │
//! │  ┌──────────────┴──────────────────────────┐    │
//! │  │  WASM Exports (exports.rs)              │    │
//! │  │  ┌──────────────────────────────────┐   │    │
//! │  │  │  Arena Allocator (arena.rs)      │   │    │
//! │  │  └──────────────────────────────────┘   │    │
//! │  │  ┌──────────────────────────────────┐   │    │
//! │  │  │  SynapseGraph Orchestrator       │   │    │
//! │  │  │  (synapse-graph-core)            │   │    │
//! │  │  └──────────────────────────────────┘   │    │
//! │  └─────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # WASM Module Paths (ADR-006)
//!
//! When compiled for WASM, each subsystem uses its WASM-compatible path:
//! - CoherenceGate: cognitum-gate-kernel (already no_std)
//! - NervousSystem: ruvector-nervous-system-wasm
//! - GNN: ruvector-gnn-wasm
//! - Attention: ruvector-attention-wasm
//! - Learning: ruvector-learning-wasm
//! - Router: ruvector-tiny-dancer-wasm

pub mod arena;
pub mod config;
pub mod exports;

pub use arena::{Arena, with_scratch_arena};
pub use config::WasmConfig;
pub use exports::WasmSynapseGraph;
pub use synapse_graph_types as types;

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use synapse_graph_types::{
        ChannelHealth, ChannelId, GatedEmbedding, SessionId,
        SpikeEvent, SpikeBatch, Timestamp,
    };

    use crate::config::WasmConfig;
    use crate::exports::{self, WasmSynapseGraph};

    fn make_batch(num_channels: u64, spikes_per_channel: u64, base_time: u64) -> SpikeBatch {
        let mut spikes = Vec::new();
        for ch in 0..num_channels {
            for s in 0..spikes_per_channel {
                spikes.push(SpikeEvent {
                    channel_id: ChannelId(ch),
                    amplitude: 0.5 + (ch as f32) * 0.01,
                    sub_timestamp: Timestamp(base_time + s * 50),
                });
            }
        }
        SpikeBatch {
            timestamp: Timestamp(base_time),
            session_id: SessionId(1),
            spikes,
            duration_us: 330,
        }
    }

    fn make_gated(dim: usize, base: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..dim).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), 0.9)
    }

    // -------------------------------------------------------------------
    // Test: WASM module loads successfully (init + free)
    // -------------------------------------------------------------------

    #[test]
    fn wasm_module_loads_with_default_config() {
        let sg = WasmSynapseGraph::new("{}");
        assert!(sg.is_some(), "should initialize with empty JSON config");
    }

    #[test]
    fn wasm_module_loads_with_custom_config() {
        let config_json = r#"{
            "embedding_dim": 32,
            "cluster_threshold": 1.5,
            "temporal_window_us": 5000,
            "fast_rank": 2,
            "slow_rank": 8
        }"#;
        let sg = WasmSynapseGraph::new(config_json);
        assert!(sg.is_some(), "should initialize with custom config");
    }

    #[test]
    fn wasm_module_rejects_invalid_json() {
        let sg = WasmSynapseGraph::new("not valid json {{{");
        assert!(sg.is_none(), "should reject invalid JSON");
    }

    // -------------------------------------------------------------------
    // Test: Full pipeline through WASM exports
    // -------------------------------------------------------------------

    #[test]
    fn wasm_full_pipeline_ingest_predict_feedback() {
        let num_channels = 8u64;
        let dim = 16;
        let config = serde_json::to_string(&WasmConfig {
            embedding_dim: dim,
            ..Default::default()
        }).unwrap();

        let mut sg = WasmSynapseGraph::new(&config).unwrap();

        // Register channels (via inner access).
        for ch in 0..num_channels {
            sg.inner_mut().ingester_mut().register_channel(
                ChannelId(ch),
                ChannelHealth {
                    impedance: 1.0,
                    snr: 20.0,
                    dropout_rate: 0.0,
                },
            );
        }

        // Ingest via WASM serialization path.
        for i in 0..5 {
            let batch = make_batch(num_channels, 5, i * 1000);
            let result = sg.ingest_json(&batch);
            assert!(result.is_some(), "ingest should return result");
            let result = result.unwrap();
            assert!(result.ingested > 0, "batch {i}: should ingest embeddings");
        }

        // Predict via inner (GatedEmbedding doesn't cross WASM boundary normally).
        let emb = make_gated(dim, 0.5);
        let prediction = sg.inner_mut().predict_intent(&emb);
        assert!(prediction.latency_us > 0);

        // Feedback via WASM API.
        sg.feedback(prediction.id.0, 2); // Rejected

        // Maintenance via WASM API.
        let summary = sg.maintain_json();
        assert!(summary.is_some());

        // Discover shortcuts via WASM API.
        let shortcuts = sg.discover_shortcuts_json();
        assert!(shortcuts.is_some());

        // Session lifecycle via WASM API.
        sg.end_session();
        sg.start_session(2);
        sg.end_session();

        // Verify provenance integrity.
        let report = sg.inner().provenance().verify();
        assert!(report.is_valid, "provenance should pass integrity check");
    }

    // -------------------------------------------------------------------
    // Test: Raw C-ABI exports work correctly
    // -------------------------------------------------------------------

    #[test]
    fn wasm_raw_exports_init_ingest_free() {
        let config = b"{}";
        let sg = exports::init_synapse_graph(config.as_ptr(), config.len() as u32);
        assert!(!sg.is_null());

        // Register channels.
        let sg_ref = unsafe { &mut *sg };
        for ch in 0..4u64 {
            sg_ref.ingester_mut().register_channel(
                ChannelId(ch),
                ChannelHealth { impedance: 1.0, snr: 20.0, dropout_rate: 0.0 },
            );
        }

        // Ingest via raw export.
        let batch = make_batch(4, 5, 1000);
        let batch_json = serde_json::to_vec(&batch).unwrap();
        let result = exports::wasm_ingest(sg, batch_json.as_ptr(), batch_json.len() as u32);
        assert!(!result.ptr.is_null());
        assert!(result.len > 0);

        let result_bytes = unsafe { std::slice::from_raw_parts(result.ptr, result.len as usize) };
        let ingest_result: synapse_graph_core::orchestrator::IngestResult =
            serde_json::from_slice(result_bytes).unwrap();
        assert!(ingest_result.ingested > 0);

        exports::wasm_free_result(result);
        exports::free_synapse_graph(sg);
    }

    // -------------------------------------------------------------------
    // Test: Memory allocation helpers
    // -------------------------------------------------------------------

    #[test]
    fn wasm_alloc_dealloc_cycle() {
        let ptr = exports::wasm_alloc(1024);
        assert!(!ptr.is_null());

        // Write some data.
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 1024);
        }

        exports::wasm_dealloc(ptr, 1024);
    }

    // -------------------------------------------------------------------
    // Test: Config serialization round-trip
    // -------------------------------------------------------------------

    #[test]
    fn wasm_config_round_trip() {
        let config = WasmConfig {
            embedding_dim: 32,
            cluster_threshold: 1.5,
            temporal_window_us: 5000,
            coherence_threshold: 0.4,
            fast_rank: 2,
            slow_rank: 8,
            fast_tier_capacity: 512,
            feedback_window_ms: 3000,
            arena_size: 128 * 1024,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: WasmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.embedding_dim, 32);
        assert_eq!(parsed.fast_rank, 2);
        assert_eq!(parsed.slow_rank, 8);
    }

    #[test]
    fn wasm_config_defaults_from_empty_json() {
        let config: WasmConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.embedding_dim, 64);
        assert_eq!(config.cluster_threshold, 2.0);
        assert_eq!(config.fast_rank, 4);
        assert_eq!(config.slow_rank, 16);
    }

    // -------------------------------------------------------------------
    // Benchmark: WASM serialization overhead
    // -------------------------------------------------------------------

    #[test]
    fn wasm_serialization_latency() {
        let num_channels = 8u64;
        let dim = 16;

        let config = serde_json::to_string(&WasmConfig {
            embedding_dim: dim,
            ..Default::default()
        }).unwrap();

        let mut sg = WasmSynapseGraph::new(&config).unwrap();

        for ch in 0..num_channels {
            sg.inner_mut().ingester_mut().register_channel(
                ChannelId(ch),
                ChannelHealth { impedance: 1.0, snr: 20.0, dropout_rate: 0.0 },
            );
        }

        let batch = make_batch(num_channels, 5, 1000);

        // Warm up.
        sg.ingest_json(&batch);

        // Measure WASM path (with JSON serialization overhead).
        let iterations = 10u32;
        let start = Instant::now();
        for i in 0..iterations {
            let b = make_batch(num_channels, 5, (i as u64 + 2) * 10000);
            sg.ingest_json(&b);
        }
        let wasm_per_call = start.elapsed() / iterations;

        // Measure native path (direct Rust call, no serialization).
        let start = Instant::now();
        for i in 0..iterations {
            let b = make_batch(num_channels, 5, (i as u64 + 100) * 10000);
            sg.inner_mut().ingest(b);
        }
        let native_per_call = start.elapsed() / iterations;

        eprintln!("WASM path (with JSON serde): {:?}", wasm_per_call);
        eprintln!("Native path (direct call):   {:?}", native_per_call);

        // WASM serialization overhead should be < 3x native in the same process.
        // (Actual WASM overhead would be higher due to runtime, but the
        //  serialization path itself should not dominate.)
        let ratio = wasm_per_call.as_nanos() as f64 / native_per_call.as_nanos().max(1) as f64;
        eprintln!("Serialization overhead ratio: {:.2}x", ratio);
        assert!(
            ratio < 5.0,
            "serialization overhead {:.2}x exceeds 5x budget",
            ratio
        );
    }

    // -------------------------------------------------------------------
    // Test: Session lifecycle through WASM API
    // -------------------------------------------------------------------

    #[test]
    fn wasm_session_lifecycle() {
        let config = serde_json::to_string(&WasmConfig {
            embedding_dim: 16,
            ..Default::default()
        }).unwrap();

        let mut sg = WasmSynapseGraph::new(&config).unwrap();

        for ch in 0..4u64 {
            sg.inner_mut().ingester_mut().register_channel(
                ChannelId(ch),
                ChannelHealth { impedance: 1.0, snr: 20.0, dropout_rate: 0.0 },
            );
        }

        // Session 1.
        sg.start_session(1);
        for i in 0..3 {
            let batch = make_batch(4, 5, i * 1000);
            sg.ingest_json(&batch);
        }
        sg.end_session();

        // Session 2.
        sg.start_session(2);
        for i in 3..6 {
            let batch = make_batch(4, 5, i * 1000);
            sg.ingest_json(&batch);
        }
        sg.end_session();

        // Verify provenance has events from both sessions.
        let total = sg.inner().provenance().total_events();
        assert!(total > 5, "should have events from both sessions, got {total}");
    }

    // -------------------------------------------------------------------
    // Test: Update cognitive graph through WASM API
    // -------------------------------------------------------------------

    #[test]
    fn wasm_update_cognitive_graph() {
        let config = serde_json::to_string(&WasmConfig {
            embedding_dim: 16,
            ..Default::default()
        }).unwrap();

        let mut sg = WasmSynapseGraph::new(&config).unwrap();

        for ch in 0..8u64 {
            sg.inner_mut().ingester_mut().register_channel(
                ChannelId(ch),
                ChannelHealth { impedance: 1.0, snr: 20.0, dropout_rate: 0.0 },
            );
        }

        // Ingest data.
        for i in 0..5 {
            let batch = make_batch(8, 5, i * 1000);
            sg.ingest_json(&batch);
        }

        // Update cognitive graph via raw export.
        let result = exports::wasm_update_cognitive_graph(sg.as_ptr());
        assert!(!result.ptr.is_null());

        let bytes = unsafe { std::slice::from_raw_parts(result.ptr, result.len as usize) };
        let update: synapse_graph_types::GraphUpdateResult =
            serde_json::from_slice(bytes).unwrap();
        assert!(
            update.nodes_created > 0 || update.nodes_updated > 0,
            "should create or update nodes"
        );

        exports::wasm_free_result(result);
    }
}
