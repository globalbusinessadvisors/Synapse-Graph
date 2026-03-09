use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::ChannelId;

/// A raw (pre-gate) embedding produced by the NervousSystemProcessor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawEmbedding {
    pub vector: Vec<f32>,
    pub channel_id: ChannelId,
}

/// BTSP learning rule outputs from a single processing pass (DDD-001).
///
/// BTSP learning is always active — there is no "inference-only" mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BtspMetadata {
    /// Average synaptic weight delta across the network for this batch.
    pub mean_weight_delta: f32,
    /// Number of synapses that were modified.
    pub synapses_modified: u32,
    /// Peak plasticity signal observed.
    pub peak_plasticity: f32,
}

/// Timing metadata extracted from a spike batch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimingMetadata {
    /// Mean inter-spike interval in microseconds.
    pub mean_isi_us: f32,
    /// Whether a burst pattern was detected.
    pub burst_detected: bool,
    /// Number of distinct burst groups found.
    pub burst_count: u32,
}

/// Output of `NervousSystemProcessor::process_spikes`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessedBatch {
    pub embeddings: Vec<RawEmbedding>,
    pub btsp_metadata: BtspMetadata,
    pub timing: TimingMetadata,
}
