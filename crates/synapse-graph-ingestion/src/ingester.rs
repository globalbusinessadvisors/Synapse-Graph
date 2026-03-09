//! SpikeIngester — aggregate root for the spike ingestion bounded context (DDD-001).
//!
//! Orchestrates the full ingestion pipeline:
//! 1. `NervousSystemProcessor::process_spikes` → `ProcessedBatch`
//! 2. For each raw embedding → `CoherenceGate::evaluate`
//! 3. Collect gated embeddings, denied events, and channel health changes.

use std::collections::HashMap;

use synapse_graph_types::{
    ChannelHealth, ChannelId, EmbeddingDenied, EmbeddingId,
    ChannelHealthChanged, GatedEmbedding, SpikeBatch,
};

use crate::gate::{CoherenceGate, GateConfig};
use crate::nervous_system::{NervousSystemConfig, NervousSystemProcessor};

/// Channel registration entry.
#[derive(Clone, Debug)]
pub struct Channel {
    pub id: ChannelId,
    pub health: ChannelHealth,
}

/// Result of a single [`SpikeIngester::ingest`] call.
#[derive(Clone, Debug)]
pub struct IngestionResult {
    pub gated: Vec<GatedEmbedding>,
    pub denied: Vec<EmbeddingDenied>,
    pub channel_health_changes: Vec<ChannelHealthChanged>,
}

/// Configuration for the [`SpikeIngester`] aggregate.
#[derive(Clone, Debug)]
pub struct IngesterConfig {
    pub gate: GateConfig,
    pub nervous_system: NervousSystemConfig,
    /// SNR threshold below which a channel is considered unhealthy.
    pub health_snr_threshold: f32,
    /// Dropout rate threshold above which a channel is considered unhealthy.
    pub health_dropout_threshold: f32,
}

impl Default for IngesterConfig {
    fn default() -> Self {
        Self {
            gate: GateConfig::default(),
            nervous_system: NervousSystemConfig::default(),
            health_snr_threshold: 2.0,
            health_dropout_threshold: 0.3,
        }
    }
}

/// The SpikeIngester aggregate root (DDD-001).
///
/// Invariant: every embedding that exits this aggregate has passed the
/// CoherenceGate. No raw embeddings escape.
pub struct SpikeIngester {
    nervous_system: NervousSystemProcessor,
    coherence_gate: CoherenceGate,
    channel_registry: HashMap<ChannelId, Channel>,
    next_embedding_id: u64,
    config: IngesterConfig,
}

impl SpikeIngester {
    pub fn new(config: IngesterConfig) -> Self {
        let coherence_gate = CoherenceGate::new(config.gate.clone());
        let nervous_system = NervousSystemProcessor::new(config.nervous_system.clone());
        Self {
            nervous_system,
            coherence_gate,
            channel_registry: HashMap::new(),
            next_embedding_id: 1,
            config,
        }
    }

    /// Register a channel with its initial health metrics.
    pub fn register_channel(&mut self, id: ChannelId, health: ChannelHealth) {
        self.coherence_gate.set_channel_health(id, health);
        self.channel_registry.insert(id, Channel { id, health });
    }

    /// Update channel health and return a change event if thresholds crossed.
    pub fn update_channel_health(
        &mut self,
        id: ChannelId,
        new_health: ChannelHealth,
    ) -> Option<ChannelHealthChanged> {
        let previous = self
            .channel_registry
            .get(&id)
            .map(|c| c.health)
            .unwrap_or(ChannelHealth {
                impedance: 1.0,
                snr: f32::MAX,
                dropout_rate: 0.0,
            });

        self.coherence_gate.set_channel_health(id, new_health);
        self.channel_registry
            .entry(id)
            .and_modify(|c| c.health = new_health)
            .or_insert(Channel {
                id,
                health: new_health,
            });

        // Emit event if health state meaningfully changed.
        let was_healthy = previous.snr >= self.config.health_snr_threshold
            && previous.dropout_rate <= self.config.health_dropout_threshold;
        let is_healthy = new_health.snr >= self.config.health_snr_threshold
            && new_health.dropout_rate <= self.config.health_dropout_threshold;

        if was_healthy != is_healthy || (previous.snr - new_health.snr).abs() > 0.5 {
            Some(ChannelHealthChanged {
                channel_id: id,
                previous,
                current: new_health,
            })
        } else {
            None
        }
    }

    /// Access the underlying coherence gate.
    pub fn coherence_gate(&self) -> &CoherenceGate {
        &self.coherence_gate
    }

    fn next_id(&mut self) -> EmbeddingId {
        let id = EmbeddingId(self.next_embedding_id);
        self.next_embedding_id += 1;
        id
    }

    /// Ingest a spike batch through the full pipeline (DDD-001).
    ///
    /// 1. Process spikes → raw embeddings via NervousSystemProcessor.
    /// 2. Gate each raw embedding through CoherenceGate.
    /// 3. Collect results and channel health changes.
    pub fn ingest(&mut self, batch: SpikeBatch) -> IngestionResult {
        let timestamp = batch.timestamp;
        let session_id = batch.session_id;

        // Step (a): NervousSystemProcessor.
        let processed = self.nervous_system.process_spikes(&batch);

        let mut gated = Vec::new();
        let mut denied = Vec::new();
        let mut channel_health_changes = Vec::new();

        // Track per-channel deny counts for health updates.
        let mut deny_counts: HashMap<ChannelId, u32> = HashMap::new();
        let mut total_counts: HashMap<ChannelId, u32> = HashMap::new();

        // Step (b): Gate each raw embedding.
        for raw_emb in processed.embeddings {
            let emb_id = self.next_id();
            let channel = raw_emb.channel_id;
            *total_counts.entry(channel).or_insert(0) += 1;

            match self
                .coherence_gate
                .evaluate(&raw_emb.vector, channel, timestamp, session_id)
            {
                synapse_graph_types::CoherenceVerdict::Permit { score } => {
                    let ge = self.coherence_gate.mint_gated(
                        raw_emb.vector,
                        channel,
                        timestamp,
                        session_id,
                        score,
                    );
                    gated.push(ge);
                }
                synapse_graph_types::CoherenceVerdict::Deny { reason } => {
                    *deny_counts.entry(channel).or_insert(0) += 1;
                    denied.push(EmbeddingDenied {
                        embedding_id: emb_id,
                        reason,
                        channel_id: channel,
                        timestamp,
                    });
                }
            }
        }

        // Step (c): Update channel health based on deny rates.
        for (&channel, &denies) in &deny_counts {
            let total = total_counts.get(&channel).copied().unwrap_or(1);
            let deny_rate = denies as f32 / total as f32;

            if let Some(current) = self.channel_registry.get(&channel) {
                let new_health = ChannelHealth {
                    impedance: current.health.impedance,
                    snr: current.health.snr * (1.0 - deny_rate),
                    dropout_rate: (current.health.dropout_rate + deny_rate) * 0.5,
                };
                if let Some(evt) = self.update_channel_health(channel, new_health) {
                    channel_health_changes.push(evt);
                }
            }
        }

        IngestionResult {
            gated,
            denied,
            channel_health_changes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{MockAdapterConfig, MockBciAdapter, MockPattern};
    use synapse_graph_types::{DenyReason, SessionId, SpikeEvent, Timestamp};

    fn default_ingester() -> SpikeIngester {
        let config = IngesterConfig {
            nervous_system: NervousSystemConfig {
                embedding_dim: 16,
                ..NervousSystemConfig::default()
            },
            ..IngesterConfig::default()
        };
        let mut ingester = SpikeIngester::new(config);
        // Register channels with healthy defaults.
        for ch in 0..8 {
            ingester.register_channel(
                ChannelId(ch),
                ChannelHealth {
                    impedance: 1.0,
                    snr: 20.0,
                    dropout_rate: 0.0,
                },
            );
        }
        ingester
    }

    #[test]
    fn clean_batch_all_gated() {
        let mut ingester = default_ingester();
        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 4,
            spikes_per_channel: 5,
            pattern: MockPattern::Constant { amplitude: 0.5 },
            ..MockAdapterConfig::default()
        });
        let batch = adapter.generate_batch();
        let result = ingester.ingest(batch);
        // All 4 channels should produce gated embeddings.
        assert_eq!(result.gated.len(), 4, "expected 4 gated embeddings");
        assert!(result.denied.is_empty(), "expected no denials");
    }

    #[test]
    fn noisy_channels_denied() {
        let mut ingester = default_ingester();

        // First ingest several clean batches to build up manifold stats.
        for i in 0..10 {
            let adapter = MockBciAdapter::new(MockAdapterConfig {
                num_channels: 4,
                spikes_per_channel: 5,
                pattern: MockPattern::Constant {
                    amplitude: 0.5 + (i as f32) * 0.001,
                },
                base_timestamp: Timestamp(i * 1000),
                ..MockAdapterConfig::default()
            });
            ingester.ingest(adapter.generate_batch());
        }

        // Now inject a batch where some channels have zero signal.
        let mut spikes = Vec::new();
        // Channels 0-1: normal
        for ch in 0..2 {
            for s in 0..5 {
                spikes.push(SpikeEvent {
                    channel_id: ChannelId(ch),
                    amplitude: 0.5,
                    sub_timestamp: Timestamp(20000 + s * 50),
                });
            }
        }
        // Channels 2-3: zero signal (will produce zero embeddings)
        for ch in 2..4 {
            for s in 0..5 {
                spikes.push(SpikeEvent {
                    channel_id: ChannelId(ch),
                    amplitude: 0.0, // zero!
                    sub_timestamp: Timestamp(20000 + s * 50),
                });
            }
        }

        let batch = SpikeBatch {
            timestamp: Timestamp(20000),
            session_id: SessionId(1),
            spikes,
            duration_us: 330,
        };
        let result = ingester.ingest(batch);
        // Zero-amplitude channels produce zero embeddings → denied by ZeroSignal check.
        assert!(
            !result.denied.is_empty(),
            "expected some denials for zero-signal channels"
        );
        for denial in &result.denied {
            assert_eq!(denial.reason, DenyReason::ZeroSignal);
        }
    }

    #[test]
    fn mock_adapter_batches_are_valid() {
        use crate::adapter::BciAdapter;

        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 16,
            spikes_per_channel: 8,
            pattern: MockPattern::Burst {
                amplitude: 1.0,
                spikes_per_burst: 4,
            },
            ..MockAdapterConfig::default()
        });
        let batch = adapter.translate(&[]).unwrap();
        assert_eq!(batch.spikes.len(), 16 * 8);
        assert_eq!(batch.duration_us, 330);
        assert_eq!(adapter.sampling_rate_hz(), 30_000);
    }

    #[test]
    fn channel_health_transition_emits_event() {
        let mut ingester = default_ingester();

        // Start healthy.
        let ch = ChannelId(0);

        // Drop SNR below threshold → should emit a health changed event.
        let evt = ingester.update_channel_health(
            ch,
            ChannelHealth {
                impedance: 1.0,
                snr: 0.5, // way below threshold of 2.0
                dropout_rate: 0.0,
            },
        );
        assert!(evt.is_some(), "expected ChannelHealthChanged event");
        let evt = evt.unwrap();
        assert_eq!(evt.channel_id, ch);
        assert!(evt.previous.snr > evt.current.snr);
    }

    #[test]
    fn ingest_latency_1024_channels() {
        let config = IngesterConfig {
            nervous_system: NervousSystemConfig {
                embedding_dim: 16,
                ..NervousSystemConfig::default()
            },
            ..IngesterConfig::default()
        };
        let mut ingester = SpikeIngester::new(config);
        for ch in 0..1024 {
            ingester.register_channel(
                ChannelId(ch),
                ChannelHealth {
                    impedance: 1.0,
                    snr: 20.0,
                    dropout_rate: 0.0,
                },
            );
        }

        let adapter = MockBciAdapter::new(MockAdapterConfig {
            num_channels: 1024,
            spikes_per_channel: 5,
            pattern: MockPattern::Constant { amplitude: 0.5 },
            ..MockAdapterConfig::default()
        });
        let batch = adapter.generate_batch();

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let b = batch.clone();
            std::hint::black_box(ingester.ingest(b));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;

        // Target: < 5ms for 1024 channels. CI VMs get generous headroom.
        let ceiling_ms = if cfg!(debug_assertions) { 200 } else { 50 };
        assert!(
            per_call.as_millis() < ceiling_ms,
            "ingest() 1024ch took {per_call:?}, expected < {ceiling_ms}ms"
        );
        eprintln!("ingest() 1024 channels: {per_call:?}");
    }
}
