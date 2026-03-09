//! BCI Adapter anti-corruption layer (DDD-001).
//!
//! Isolates the domain from vendor-specific BCI hardware formats.

use synapse_graph_types::{AdapterError, ChannelId, SessionId, SpikeBatch, SpikeEvent, Timestamp};

/// Anti-corruption layer translating raw BCI device bytes into domain types.
pub trait BciAdapter: Send + Sync {
    /// Translate a raw byte buffer from the BCI device into a [`SpikeBatch`].
    fn translate(&self, raw_data: &[u8]) -> Result<SpikeBatch, AdapterError>;

    /// The set of channel IDs this adapter produces.
    fn channel_map(&self) -> &[ChannelId];

    /// The sampling rate of the underlying hardware in Hz.
    fn sampling_rate_hz(&self) -> u32;
}

/// Pattern for synthetic spike generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MockPattern {
    /// All channels emit a constant amplitude.
    Constant { amplitude: f32 },
    /// Channels emit periodic bursts of high-amplitude spikes.
    Burst {
        amplitude: f32,
        spikes_per_burst: usize,
    },
    /// Amplitudes drift linearly across channels.
    Drift {
        base_amplitude: f32,
        drift_per_channel: f32,
    },
    /// Random-ish noise (deterministic, seeded by channel index).
    Noise { scale: f32 },
}

/// Configuration for [`MockBciAdapter`].
#[derive(Clone, Debug)]
pub struct MockAdapterConfig {
    pub num_channels: u64,
    pub spikes_per_channel: usize,
    pub pattern: MockPattern,
    pub sampling_rate_hz: u32,
    pub session_id: SessionId,
    pub base_timestamp: Timestamp,
    pub duration_us: u32,
}

impl Default for MockAdapterConfig {
    fn default() -> Self {
        Self {
            num_channels: 32,
            spikes_per_channel: 10,
            pattern: MockPattern::Constant { amplitude: 0.5 },
            sampling_rate_hz: 30_000,
            session_id: SessionId(1),
            base_timestamp: Timestamp(0),
            duration_us: 330, // ~0.33ms at 30kHz
        }
    }
}

/// Mock BCI adapter that generates synthetic spike batches for testing.
///
/// This is the spike-train simulator from SPARC Phase 1 Task 6.
pub struct MockBciAdapter {
    config: MockAdapterConfig,
    channels: Vec<ChannelId>,
}

impl MockBciAdapter {
    pub fn new(config: MockAdapterConfig) -> Self {
        let channels: Vec<ChannelId> = (0..config.num_channels).map(ChannelId).collect();
        Self { config, channels }
    }

    /// Generate a synthetic [`SpikeBatch`] directly (bypasses the byte layer).
    pub fn generate_batch(&self) -> SpikeBatch {
        let mut spikes = Vec::with_capacity(
            self.config.num_channels as usize * self.config.spikes_per_channel,
        );

        for ch_idx in 0..self.config.num_channels {
            let channel_id = ChannelId(ch_idx);
            for s in 0..self.config.spikes_per_channel {
                let amplitude = match self.config.pattern {
                    MockPattern::Constant { amplitude } => amplitude,
                    MockPattern::Burst {
                        amplitude,
                        spikes_per_burst,
                    } => {
                        if s < spikes_per_burst {
                            amplitude
                        } else {
                            amplitude * 0.1
                        }
                    }
                    MockPattern::Drift {
                        base_amplitude,
                        drift_per_channel,
                    } => base_amplitude + (ch_idx as f32) * drift_per_channel,
                    MockPattern::Noise { scale } => {
                        // Deterministic pseudo-random based on channel + spike index.
                        let seed = (ch_idx as u32).wrapping_mul(2654435761)
                            ^ (s as u32).wrapping_mul(340573321);
                        let norm = (seed % 10000) as f32 / 10000.0;
                        norm * scale
                    }
                };

                let sub_timestamp = Timestamp(
                    self.config.base_timestamp.0
                        + (s as u64) * (self.config.duration_us as u64)
                            / (self.config.spikes_per_channel.max(1) as u64),
                );

                spikes.push(SpikeEvent {
                    channel_id,
                    amplitude,
                    sub_timestamp,
                });
            }
        }

        SpikeBatch {
            timestamp: self.config.base_timestamp,
            session_id: self.config.session_id,
            spikes,
            duration_us: self.config.duration_us,
        }
    }
}

impl BciAdapter for MockBciAdapter {
    fn translate(&self, _raw_data: &[u8]) -> Result<SpikeBatch, AdapterError> {
        // The mock adapter ignores the raw bytes and generates synthetic data.
        Ok(self.generate_batch())
    }

    fn channel_map(&self) -> &[ChannelId] {
        &self.channels
    }

    fn sampling_rate_hz(&self) -> u32 {
        self.config.sampling_rate_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_adapter_produces_correct_batch_shape() {
        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 8,
            spikes_per_channel: 5,
            ..MockAdapterConfig::default()
        });
        let batch = adapter.generate_batch();
        assert_eq!(batch.spikes.len(), 8 * 5);
        assert_eq!(batch.session_id, SessionId(1));
    }

    #[test]
    fn mock_adapter_translate_returns_ok() {
        let adapter = MockBciAdapter::new(MockAdapterConfig::default());
        let result = adapter.translate(&[0u8; 64]);
        assert!(result.is_ok());
        let batch = result.unwrap();
        assert_eq!(
            batch.spikes.len(),
            (32 * 10) as usize // default: 32 channels × 10 spikes
        );
    }

    #[test]
    fn mock_adapter_channel_map() {
        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 4,
            ..MockAdapterConfig::default()
        });
        let map = adapter.channel_map();
        assert_eq!(map.len(), 4);
        assert_eq!(map[0], ChannelId(0));
        assert_eq!(map[3], ChannelId(3));
    }

    #[test]
    fn mock_adapter_burst_pattern() {
        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 1,
            spikes_per_channel: 6,
            pattern: MockPattern::Burst {
                amplitude: 1.0,
                spikes_per_burst: 3,
            },
            ..MockAdapterConfig::default()
        });
        let batch = adapter.generate_batch();
        // First 3 spikes should be high amplitude, rest low.
        assert_eq!(batch.spikes[0].amplitude, 1.0);
        assert_eq!(batch.spikes[2].amplitude, 1.0);
        assert!((batch.spikes[3].amplitude - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn mock_adapter_drift_pattern() {
        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 4,
            spikes_per_channel: 1,
            pattern: MockPattern::Drift {
                base_amplitude: 0.5,
                drift_per_channel: 0.1,
            },
            ..MockAdapterConfig::default()
        });
        let batch = adapter.generate_batch();
        // Channel 0 → 0.5, channel 1 → 0.6, channel 2 → 0.7, channel 3 → 0.8
        assert!((batch.spikes[0].amplitude - 0.5).abs() < f32::EPSILON);
        assert!((batch.spikes[3].amplitude - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn mock_adapter_noise_pattern() {
        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 2,
            spikes_per_channel: 3,
            pattern: MockPattern::Noise { scale: 2.0 },
            ..MockAdapterConfig::default()
        });
        let batch = adapter.generate_batch();
        // All amplitudes should be in [0, scale).
        for spike in &batch.spikes {
            assert!(spike.amplitude >= 0.0 && spike.amplitude < 2.0);
        }
    }
}
