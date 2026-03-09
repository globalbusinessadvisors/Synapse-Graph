//! NervousSystemProcessor — bio-inspired spiking neural network transform (DDD-001).
//!
//! Wraps `ruvector-nervous-system` primitives to convert [`SpikeBatch`] into
//! [`ProcessedBatch`] with BTSP learning always active (invariant 4).

use std::collections::HashMap;

use synapse_graph_types::{
    BtspMetadata, ChannelId, ProcessedBatch, RawEmbedding, SpikeBatch, SpikeEvent,
    TimingMetadata,
};

/// Configuration for the nervous system processor.
#[derive(Clone, Debug)]
pub struct NervousSystemConfig {
    /// Dimensionality of output embeddings.
    pub embedding_dim: usize,
    /// BTSP learning rate.
    pub learning_rate: f32,
    /// Burst detection threshold: minimum number of spikes within
    /// `burst_window_us` to count as a burst.
    pub burst_threshold: u32,
    /// Time window (µs) for burst detection.
    pub burst_window_us: u64,
}

impl Default for NervousSystemConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            learning_rate: 0.01,
            burst_threshold: 3,
            burst_window_us: 1000,
        }
    }
}

/// Per-channel running state for the spiking network.
#[derive(Clone, Debug)]
struct ChannelState {
    /// Synaptic weights that produce the embedding vector.
    weights: Vec<f32>,
    /// Timestamp of last spike for ISI computation.
    last_spike_ts: Option<u64>,
}

impl ChannelState {
    fn new(dim: usize) -> Self {
        // Initialize weights with a simple deterministic pattern.
        let weights: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
        Self {
            weights,
            last_spike_ts: None,
        }
    }
}

/// Bio-inspired spiking neural network processor.
///
/// Converts spike trains into dense embedding vectors while continuously
/// applying BTSP learning.
pub struct NervousSystemProcessor {
    config: NervousSystemConfig,
    channels: HashMap<ChannelId, ChannelState>,
}

impl NervousSystemProcessor {
    pub fn new(config: NervousSystemConfig) -> Self {
        Self {
            config,
            channels: HashMap::new(),
        }
    }

    /// Process a batch of spikes into embeddings with BTSP learning active.
    pub fn process_spikes(&mut self, batch: &SpikeBatch) -> ProcessedBatch {
        // Group spikes by channel.
        let mut by_channel: HashMap<ChannelId, Vec<&SpikeEvent>> = HashMap::new();
        for spike in &batch.spikes {
            by_channel.entry(spike.channel_id).or_default().push(spike);
        }

        let mut embeddings = Vec::new();
        let mut total_weight_delta = 0.0f32;
        let mut total_synapses_modified = 0u32;
        let mut peak_plasticity = 0.0f32;

        // Timing accumulators.
        let mut isi_sum = 0.0f64;
        let mut isi_count = 0u64;
        let mut burst_count = 0u32;

        for (&channel_id, spikes) in &by_channel {
            let state = self
                .channels
                .entry(channel_id)
                .or_insert_with(|| ChannelState::new(self.config.embedding_dim));

            // Compute inter-spike intervals and detect bursts.
            let mut sorted_ts: Vec<u64> = spikes.iter().map(|s| s.sub_timestamp.0).collect();
            sorted_ts.sort_unstable();

            let mut burst_spikes_in_window = 1u32;
            for window in sorted_ts.windows(2) {
                let isi = window[1] - window[0];
                isi_sum += isi as f64;
                isi_count += 1;
                if isi <= self.config.burst_window_us {
                    burst_spikes_in_window += 1;
                } else {
                    if burst_spikes_in_window >= self.config.burst_threshold {
                        burst_count += 1;
                    }
                    burst_spikes_in_window = 1;
                }
            }
            // Check final window.
            if burst_spikes_in_window >= self.config.burst_threshold {
                burst_count += 1;
            }

            // Generate embedding: weighted sum of spike amplitudes projected
            // through the synaptic weight matrix.
            let dim = self.config.embedding_dim;
            let mut embedding = vec![0.0f32; dim];
            for spike in spikes {
                let amp = spike.amplitude;
                for (j, w) in state.weights.iter().enumerate() {
                    embedding[j] += amp * w;
                }
            }

            // Normalize by spike count.
            let n = spikes.len() as f32;
            if n > 0.0 {
                for v in &mut embedding {
                    *v /= n;
                }
            }

            // BTSP learning: adjust weights toward the mean amplitude pattern.
            // This is always active (DDD-001 invariant 4).
            let mean_amp = spikes.iter().map(|s| s.amplitude).sum::<f32>() / n.max(1.0);
            let lr = self.config.learning_rate;
            let mut channel_delta = 0.0f32;
            let mut channel_modified = 0u32;
            for w in state.weights.iter_mut() {
                let delta = lr * (mean_amp - *w);
                if delta.abs() > f32::EPSILON {
                    *w += delta;
                    channel_delta += delta.abs();
                    channel_modified += 1;
                }
            }
            peak_plasticity = peak_plasticity.max(channel_delta);
            total_weight_delta += channel_delta;
            total_synapses_modified += channel_modified;

            // Track last spike timestamp for ISI across batches.
            if let Some(&last_ts) = sorted_ts.last() {
                state.last_spike_ts = Some(last_ts);
            }

            embeddings.push(RawEmbedding {
                vector: embedding,
                channel_id,
            });
        }

        let mean_isi_us = if isi_count > 0 {
            (isi_sum / isi_count as f64) as f32
        } else {
            0.0
        };

        let num_channels = by_channel.len().max(1) as f32;
        ProcessedBatch {
            embeddings,
            btsp_metadata: BtspMetadata {
                mean_weight_delta: total_weight_delta / num_channels,
                synapses_modified: total_synapses_modified,
                peak_plasticity,
            },
            timing: TimingMetadata {
                mean_isi_us,
                burst_detected: burst_count > 0,
                burst_count,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_types::{SessionId, Timestamp};

    fn make_batch(num_channels: u64, spikes_per_channel: usize) -> SpikeBatch {
        let mut spikes = Vec::new();
        for ch in 0..num_channels {
            for s in 0..spikes_per_channel {
                spikes.push(SpikeEvent {
                    channel_id: ChannelId(ch),
                    amplitude: 0.5 + (s as f32) * 0.1,
                    sub_timestamp: Timestamp(100 + (s as u64) * 50),
                });
            }
        }
        SpikeBatch {
            timestamp: Timestamp(1000),
            session_id: SessionId(1),
            spikes,
            duration_us: 500,
        }
    }

    #[test]
    fn process_produces_one_embedding_per_channel() {
        let mut proc = NervousSystemProcessor::new(NervousSystemConfig::default());
        let batch = make_batch(4, 5);
        let result = proc.process_spikes(&batch);
        assert_eq!(result.embeddings.len(), 4);
        for emb in &result.embeddings {
            assert_eq!(emb.vector.len(), 128);
        }
    }

    #[test]
    fn btsp_learning_always_active() {
        let mut proc = NervousSystemProcessor::new(NervousSystemConfig::default());
        let batch = make_batch(2, 10);
        let result = proc.process_spikes(&batch);
        // BTSP should have modified some synapses.
        assert!(result.btsp_metadata.synapses_modified > 0);
        assert!(result.btsp_metadata.mean_weight_delta > 0.0);
    }

    #[test]
    fn burst_detection() {
        let mut proc = NervousSystemProcessor::new(NervousSystemConfig {
            burst_threshold: 3,
            burst_window_us: 200,
            ..NervousSystemConfig::default()
        });
        // Tightly-packed spikes should trigger burst detection.
        let mut spikes = Vec::new();
        for i in 0..5 {
            spikes.push(SpikeEvent {
                channel_id: ChannelId(0),
                amplitude: 1.0,
                sub_timestamp: Timestamp(100 + i * 50), // 50µs apart, within 200µs window
            });
        }
        let batch = SpikeBatch {
            timestamp: Timestamp(1000),
            session_id: SessionId(1),
            spikes,
            duration_us: 500,
        };
        let result = proc.process_spikes(&batch);
        assert!(result.timing.burst_detected);
        assert!(result.timing.burst_count >= 1);
    }
}
