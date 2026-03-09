//! SynapseGraph — application orchestrator wiring all bounded contexts (SPARC).
//!
//! This is the top-level entry point that implements the full data flow:
//! ingest → temporal → memory → cognitive → prediction → adaptation → provenance.

use serde::{Deserialize, Serialize};

use synapse_graph_types::{
    AdaptationApplied, AdaptationConfig, AdaptationType, CognitiveShortcut, Consolidated,
    EmbeddingIngested, GatedEmbedding, GraphHealed, GraphTopologyChanged, GraphUpdateResult,
    IntentPrediction, LoraTarget, NeuralEvent, PredictionFeedback, PredictionId, PredictionMade,
    SessionId, SpikeBatch, TemporalMeta, TierWeights, TimingMetadata,
};

use synapse_graph_adaptation::engine::AdaptationEngine;
use synapse_graph_cognitive::graph::{CognitiveGraph, CognitiveGraphConfig};
use synapse_graph_ingestion::ingester::{IngesterConfig, SpikeIngester};
use synapse_graph_memory::store::{VectorMemory, VectorMemoryConfig};
use synapse_graph_prediction::predictor::{IntentPredictor, PredictionConfig};
use synapse_graph_provenance::log::ProvenanceLog;
use synapse_graph_temporal::learner::{TemporalConfig, TemporalLearner};

/// Top-level configuration for the SynapseGraph orchestrator.
#[derive(Clone, Debug)]
pub struct SynapseGraphConfig {
    pub ingester: IngesterConfig,
    pub memory: VectorMemoryConfig,
    pub temporal: TemporalConfig,
    pub cognitive: CognitiveGraphConfig,
    pub prediction: PredictionConfig,
    pub adaptation: AdaptationConfig,
    /// Threshold for clustering recent embeddings. Default: 2.0.
    pub cluster_threshold: f32,
    /// Temporal adjacency window in microseconds. Default: 10_000 (10ms).
    pub temporal_window_us: u64,
    /// Default tier weights for context blending.
    pub tier_weights: TierWeights,
}

impl Default for SynapseGraphConfig {
    fn default() -> Self {
        Self {
            ingester: IngesterConfig::default(),
            memory: VectorMemoryConfig::default(),
            temporal: TemporalConfig::default(),
            cognitive: CognitiveGraphConfig::default(),
            prediction: PredictionConfig::default(),
            adaptation: AdaptationConfig::default(),
            cluster_threshold: 2.0,
            temporal_window_us: 10_000,
            tier_weights: TierWeights::default(),
        }
    }
}

/// Result of a single ingest cycle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestResult {
    pub ingested: usize,
    pub denied: usize,
    pub stored: usize,
}

/// Result of a maintenance cycle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaintenanceSummary {
    pub patterns_promoted: usize,
    pub weak_points_healed: usize,
    pub deltas_consolidated: usize,
}

/// The SynapseGraph application orchestrator.
///
/// Wires all bounded contexts together per the SPARC pseudocode and DDD context map.
pub struct SynapseGraph {
    ingester: SpikeIngester,
    memory: VectorMemory,
    temporal: TemporalLearner,
    cognitive: CognitiveGraph,
    predictor: IntentPredictor,
    adaptation: AdaptationEngine,
    provenance: ProvenanceLog,
    config: SynapseGraphConfig,
    current_session: SessionId,
}

impl SynapseGraph {
    /// Initialize the SynapseGraph per SPARC initialize_synapse_graph().
    pub fn new(config: SynapseGraphConfig) -> Self {
        let session = SessionId(1);
        Self {
            ingester: SpikeIngester::new(config.ingester.clone()),
            memory: VectorMemory::new(config.memory.clone()),
            temporal: TemporalLearner::new(config.temporal.clone()),
            cognitive: CognitiveGraph::new(config.cognitive.clone()),
            predictor: IntentPredictor::new(config.prediction.clone()),
            adaptation: AdaptationEngine::new(config.adaptation.clone()),
            provenance: ProvenanceLog::new(session),
            config,
            current_session: session,
        }
    }

    /// Main ingestion pipeline per SPARC ingest_spike_train().
    pub fn ingest(&mut self, batch: SpikeBatch) -> IngestResult {
        let _session_id = batch.session_id;
        let _batch_timestamp = batch.timestamp;

        // (a) Spike processing + coherence gating.
        let result = self.ingester.ingest(batch);

        // (b) Log denied embeddings to provenance.
        for denied in &result.denied {
            self.provenance
                .append(NeuralEvent::Denied(denied.clone()));
        }

        // (c) For each gated embedding:
        let mut stored_count = 0;
        for gated in &result.gated {
            // (c1) Compute timing metadata for temporal classification.
            let timing = TimingMetadata {
                mean_isi_us: 1000.0,   // default timing
                burst_detected: false,
                burst_count: 0,
            };
            let tier = self.temporal.router().classify(gated, &timing);
            let meta = TemporalMeta {
                tier,
                timestamp: gated.timestamp(),
                channel_id: gated.channel_id(),
                session_id: gated.session_id(),
                coherence_score: gated.coherence_score(),
            };

            // (c2) Store in vector memory.
            let insert_emb = gated.clone();
            if let Ok((emb_id, _stored_event)) = self.memory.insert(insert_emb, meta.clone()) {
                stored_count += 1;

                // (c3) Accumulate in temporal learner.
                self.temporal.accumulate(gated, &timing);

                // (c4) Log ingestion to provenance.
                self.provenance.append(NeuralEvent::Ingested(
                    EmbeddingIngested {
                        embedding_id: emb_id,
                        gated_embedding: gated.clone(),
                    },
                ));
            }
        }

        // (d) Log channel health changes.
        for change in &result.channel_health_changes {
            self.provenance
                .append(NeuralEvent::ChannelHealth(change.clone()));
        }

        IngestResult {
            ingested: result.gated.len(),
            denied: result.denied.len(),
            stored: stored_count,
        }
    }

    /// Update the cognitive graph from recent memory state.
    pub fn update_cognitive_graph(&mut self) -> GraphUpdateResult {
        // (a) Cluster recent embeddings.
        let clusters = self.memory.cluster_recent(self.config.cluster_threshold);

        // (b) Extract temporal sequences.
        let sequences: Vec<_> = clusters
            .iter()
            .flat_map(|c| self.temporal.extract_sequences(c))
            .collect();

        // (c) Find temporally adjacent pairs.
        let neighbors: Vec<_> = clusters
            .iter()
            .flat_map(|c| self.memory.find_temporally_adjacent(c, self.config.temporal_window_us))
            .collect();

        // (d) Update the cognitive graph.
        let result = self.cognitive.update(clusters, sequences, neighbors);

        // (e) Log topology change to provenance.
        let topo = self.cognitive.topology();
        self.provenance.append(NeuralEvent::TopologyChanged(
            GraphTopologyChanged {
                total_nodes: topo.total_nodes,
                total_edges: topo.total_edges,
                average_min_cut: topo.average_min_cut,
                shortcut_count: topo.shortcut_count,
            },
        ));

        result
    }

    /// Discover cognitive shortcuts via GNN engine.
    pub fn discover_shortcuts(&mut self) -> Vec<CognitiveShortcut> {
        let (shortcuts, events, _failures) = self.cognitive.discover_shortcuts();

        for event in &events {
            self.provenance.append(NeuralEvent::ShortcutFound(
                event.clone(),
            ));
        }

        shortcuts
    }

    /// Run intent prediction for a partial pattern.
    pub fn predict_intent(&mut self, partial: &GatedEmbedding) -> IntentPrediction {
        let ctx = self.temporal.get_context(self.config.tier_weights);

        let (prediction, cold_events) = self.predictor.predict(
            partial,
            &ctx,
            &self.memory,
            self.cognitive.shortcuts(),
            self.cognitive.hypergraph(),
        );

        // Log cold start events.
        for event in &cold_events {
            self.provenance.append(NeuralEvent::NewIntent(event.clone()));
        }

        // Log prediction to provenance.
        self.provenance.append(NeuralEvent::Predicted(PredictionMade {
            prediction: prediction.clone(),
            feedback_window_ms: self.predictor.config().feedback_window_ms,
        }));

        prediction
    }

    /// Provide feedback on a prediction and trigger adaptation if needed.
    pub fn provide_feedback(
        &mut self,
        prediction_id: PredictionId,
        feedback: PredictionFeedback,
    ) {
        if let Some(signal) = self.predictor.record_feedback(prediction_id, feedback) {
            let result = self.adaptation.adapt(signal);

            // Propagate LoRA deltas to GNN and attention scorer.
            if let Some(delta) = self.adaptation.last_delta() {
                let _ = self.cognitive.gnn_engine_mut().apply_lora_delta(delta);
                let _ = self.predictor.scorer_mut().apply_lora_delta(delta);
            }

            // Log to provenance.
            self.provenance.append(NeuralEvent::Adapted(AdaptationApplied {
                result: result.clone(),
                affected_layers: result.delta.as_ref()
                    .map(|d| vec![d.layer_id])
                    .unwrap_or_default(),
                trigger: result.delta.as_ref()
                    .map(|d| d.trigger.clone())
                    .unwrap_or(AdaptationType::FastCorrection),
            }));
        }
    }

    /// Run the maintenance cycle: temporal promotions, graph healing, LoRA consolidation.
    pub fn maintain(&mut self) -> MaintenanceSummary {
        // (a) Temporal tier maintenance.
        let temporal_result = self.temporal.maintain();

        // (b) Graph self-healing.
        let weak_points = self.cognitive.analyze_weak_points();
        let healing = self.cognitive.heal(weak_points, &[]);

        self.provenance.append(NeuralEvent::Healed(GraphHealed {
            weak_points_reinforced: healing.weak_points_reinforced,
            dead_regions_pruned: healing.dead_regions_pruned,
            nodes_removed: healing.nodes_removed,
            edges_added: healing.edges_added,
            edges_removed: healing.edges_removed,
        }));

        // (c) Slow LoRA consolidation.
        let consolidation = self.adaptation.consolidate();

        self.provenance.append(NeuralEvent::Consolidated(Consolidated {
            result: consolidation.clone(),
            accumulated_fast_deltas_count: consolidation.deltas_consolidated,
        }));

        MaintenanceSummary {
            patterns_promoted: temporal_result.promoted,
            weak_points_healed: healing.weak_points_reinforced,
            deltas_consolidated: consolidation.deltas_consolidated,
        }
    }

    /// Start a new session.
    pub fn start_session(&mut self, session_id: SessionId) {
        self.current_session = session_id;
        self.provenance.set_session(session_id);
    }

    /// End the current session.
    pub fn end_session(&mut self) {
        // Archive medium tier session tensor.
        self.temporal.end_session();

        // Final consolidation pass.
        let consolidation = self.adaptation.consolidate();
        if consolidation.deltas_consolidated > 0 {
            self.provenance.append(NeuralEvent::Consolidated(Consolidated {
                result: consolidation.clone(),
                accumulated_fast_deltas_count: consolidation.deltas_consolidated,
            }));
        }
    }

    // -----------------------------------------------------------------------
    // Accessors for testing
    // -----------------------------------------------------------------------

    pub fn ingester(&self) -> &SpikeIngester {
        &self.ingester
    }

    pub fn ingester_mut(&mut self) -> &mut SpikeIngester {
        &mut self.ingester
    }

    pub fn memory(&self) -> &VectorMemory {
        &self.memory
    }

    pub fn temporal(&self) -> &TemporalLearner {
        &self.temporal
    }

    pub fn cognitive(&self) -> &CognitiveGraph {
        &self.cognitive
    }

    pub fn predictor(&self) -> &IntentPredictor {
        &self.predictor
    }

    pub fn adaptation(&self) -> &AdaptationEngine {
        &self.adaptation
    }

    pub fn provenance(&self) -> &ProvenanceLog {
        &self.provenance
    }
}
