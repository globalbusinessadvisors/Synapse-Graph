//! WASM-compatible configuration types with JSON serialization.
//!
//! These types bridge the gap between the JSON config provided by the
//! host runtime (JavaScript/WASI) and the internal Rust config structs
//! which don't derive Serialize/Deserialize.

use serde::{Deserialize, Serialize};

use synapse_graph_cognitive::graph::CognitiveGraphConfig;
use synapse_graph_core::orchestrator::SynapseGraphConfig;
use synapse_graph_ingestion::ingester::IngesterConfig;
use synapse_graph_ingestion::nervous_system::NervousSystemConfig;
use synapse_graph_memory::store::VectorMemoryConfig;
use synapse_graph_prediction::predictor::PredictionConfig;
use synapse_graph_temporal::learner::TemporalConfig;
use synapse_graph_types::AdaptationConfig;

/// JSON-serializable configuration for the WASM SynapseGraph module.
///
/// All fields are optional and default to sensible values per ADR-006.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WasmConfig {
    /// Embedding dimension (default: 64).
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// Cluster threshold for cognitive graph updates (default: 2.0).
    #[serde(default = "default_cluster_threshold")]
    pub cluster_threshold: f32,

    /// Temporal adjacency window in microseconds (default: 10000).
    #[serde(default = "default_temporal_window_us")]
    pub temporal_window_us: u64,

    /// Coherence gate threshold (default: 0.3).
    #[serde(default = "default_coherence_threshold")]
    pub coherence_threshold: f32,

    /// Fast LoRA rank (default: 4).
    #[serde(default = "default_fast_rank")]
    pub fast_rank: usize,

    /// Slow LoRA rank (default: 16).
    #[serde(default = "default_slow_rank")]
    pub slow_rank: usize,

    /// Fast-tier ring buffer capacity (default: 1024).
    #[serde(default = "default_fast_tier_capacity")]
    pub fast_tier_capacity: usize,

    /// Feedback window in milliseconds (default: 5000).
    #[serde(default = "default_feedback_window_ms")]
    pub feedback_window_ms: u64,

    /// Arena size in bytes for WASM memory management (default: 262144 = 256KB).
    #[serde(default = "default_arena_size")]
    pub arena_size: usize,
}

fn default_embedding_dim() -> usize { 64 }
fn default_cluster_threshold() -> f32 { 2.0 }
fn default_temporal_window_us() -> u64 { 10_000 }
fn default_coherence_threshold() -> f32 { 0.3 }
fn default_fast_rank() -> usize { 4 }
fn default_slow_rank() -> usize { 16 }
fn default_fast_tier_capacity() -> usize { 1024 }
fn default_feedback_window_ms() -> u64 { 5000 }
fn default_arena_size() -> usize { 256 * 1024 }

impl WasmConfig {
    /// Convert the WASM config into the internal SynapseGraphConfig.
    pub fn into_synapse_config(self) -> SynapseGraphConfig {
        SynapseGraphConfig {
            ingester: IngesterConfig {
                nervous_system: NervousSystemConfig {
                    embedding_dim: self.embedding_dim,
                    ..Default::default()
                },
                ..Default::default()
            },
            memory: VectorMemoryConfig {
                dimension: self.embedding_dim,
                ..Default::default()
            },
            temporal: TemporalConfig::default(),
            cognitive: CognitiveGraphConfig::default(),
            prediction: PredictionConfig::default(),
            adaptation: AdaptationConfig {
                fast_rank: self.fast_rank,
                slow_rank: self.slow_rank,
                ..Default::default()
            },
            cluster_threshold: self.cluster_threshold,
            temporal_window_us: self.temporal_window_us,
            ..Default::default()
        }
    }
}
