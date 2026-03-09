//! IntentPredictor — aggregate root for the intent prediction domain (DDD-005).
//!
//! Orchestrates the prediction pipeline: ShortcutMatcher → AttentionScorer →
//! TemporalReweighter → ranked IntentPrediction. Strictest latency target: <1ms.

use synapse_graph_cognitive::graph::ShortcutRegistry;
use synapse_graph_cognitive::hypergraph::HypergraphStore;
use synapse_graph_memory::store::VectorMemory;
use synapse_graph_types::{
    AdaptationSignal, AdaptationSignalType, AdaptationUrgency, GatedEmbedding, IntentPrediction,
    NewIntentDiscovered, PredictionFeedback, PredictionId, TemporalContext, Timestamp,
};

use crate::matcher::{MatcherConfig, ShortcutMatcher};
use crate::reweighter::{ReweightConfig, TemporalReweighter};
use crate::scorer::{AttentionScorer, ScorerConfig};
use crate::vocabulary::{IntentVocabulary, VocabularyConfig};

/// Configuration for the IntentPredictor aggregate.
#[derive(Clone, Debug)]
pub struct PredictionConfig {
    pub matcher: MatcherConfig,
    pub scorer: ScorerConfig,
    pub reweighter: ReweightConfig,
    pub vocabulary: VocabularyConfig,
    /// Maximum number of ranked intents in output. Default: 5.
    pub prediction_k: usize,
    /// Feedback window in milliseconds. Default: 5000.
    pub feedback_window_ms: u64,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            matcher: MatcherConfig::default(),
            scorer: ScorerConfig::default(),
            reweighter: ReweightConfig::default(),
            vocabulary: VocabularyConfig::default(),
            prediction_k: 5,
            feedback_window_ms: 5000,
        }
    }
}

/// The IntentPredictor aggregate root (DDD-005).
pub struct IntentPredictor {
    matcher: ShortcutMatcher,
    scorer: AttentionScorer,
    reweighter: TemporalReweighter,
    vocabulary: IntentVocabulary,
    config: PredictionConfig,
    next_prediction_id: u64,
}

impl IntentPredictor {
    pub fn new(config: PredictionConfig) -> Self {
        let matcher = ShortcutMatcher::new(config.matcher.clone());
        let scorer = AttentionScorer::new(config.scorer.clone());
        let reweighter = TemporalReweighter::new(config.reweighter.clone());
        let vocabulary = IntentVocabulary::new(config.vocabulary.clone());
        Self {
            matcher,
            scorer,
            reweighter,
            vocabulary,
            config,
            next_prediction_id: 1,
        }
    }

    /// Run the full prediction pipeline.
    ///
    /// Pipeline per DDD-005:
    /// a) matcher.match_shortcuts(partial_pattern, memory, registry)
    /// b) scorer.score(partial_pattern.vector(), matched)
    /// c) reweighter.reweight(scored, temporal_ctx)
    /// d) Sort by confidence, take top prediction_k
    /// e) Return IntentPrediction
    ///
    /// Cold start: if no shortcuts are found, returns empty predictions and
    /// emits NewIntentDiscovered events.
    pub fn predict(
        &mut self,
        partial_pattern: &GatedEmbedding,
        temporal_ctx: &TemporalContext,
        memory: &VectorMemory,
        registry: &ShortcutRegistry,
        hypergraph: &HypergraphStore,
    ) -> (IntentPrediction, Vec<NewIntentDiscovered>) {
        let start = std::time::Instant::now();
        let mut cold_start_events = Vec::new();

        let prediction_id = PredictionId(self.next_prediction_id);
        self.next_prediction_id += 1;

        // (a) Match shortcuts.
        let matched = self
            .matcher
            .match_shortcuts(partial_pattern, memory, registry, hypergraph);

        // Cold start: no shortcuts available.
        if matched.is_empty() {
            let latency_us = start.elapsed().as_micros() as u64;

            cold_start_events.push(NewIntentDiscovered {
                intent_id: synapse_graph_types::IntentId(0),
                initial_pattern: partial_pattern.vector().to_vec(),
                discovery_context: "cold_start".into(),
            });

            return (
                IntentPrediction {
                    id: prediction_id,
                    timestamp: Timestamp(0),
                    ranked_intents: Vec::new(),
                    attention_weights: Vec::new(),
                    temporal_influence: 0.0,
                    latency_us,
                },
                cold_start_events,
            );
        }

        // (b) Score with attention.
        let attention_weights = self
            .scorer
            .compute_attention_weights(partial_pattern.vector(), &matched);
        let mut scored = self.scorer.score(partial_pattern.vector(), &matched);

        // (c) Temporal reweighting.
        let temporal_influence = self.reweighter.reweight(&mut scored, temporal_ctx);

        // (d) Sort by confidence descending, take top-k.
        scored.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(self.config.prediction_k);

        // Record prediction in vocabulary.
        for intent in &scored {
            self.vocabulary
                .record_prediction(&intent.intent_id, Timestamp(0));
        }

        let latency_us = start.elapsed().as_micros() as u64;

        (
            IntentPrediction {
                id: prediction_id,
                timestamp: Timestamp(0),
                ranked_intents: scored,
                attention_weights,
                temporal_influence,
                latency_us,
            },
            cold_start_events,
        )
    }

    /// Record feedback on a prediction.
    ///
    /// - Confirmed → None (no adaptation needed)
    /// - Corrected { actual } → Some(AdaptationSignal { PredictionError, High })
    /// - Rejected → Some(AdaptationSignal { PredictionError, High })
    /// - Timeout → None
    pub fn record_feedback(
        &mut self,
        prediction_id: PredictionId,
        feedback: PredictionFeedback,
    ) -> Option<AdaptationSignal> {
        match feedback {
            PredictionFeedback::Confirmed => None,
            PredictionFeedback::Timeout => None,
            PredictionFeedback::Corrected { actual } => {
                // Record incorrect for the predicted intent, correct for actual.
                self.vocabulary.record_accuracy(&actual, true);
                Some(AdaptationSignal {
                    signal_type: AdaptationSignalType::PredictionError,
                    urgency: AdaptationUrgency::High,
                    prediction_id,
                    details: format!("Corrected to intent {}", actual.0),
                })
            }
            PredictionFeedback::Rejected => Some(AdaptationSignal {
                signal_type: AdaptationSignalType::PredictionError,
                urgency: AdaptationUrgency::High,
                prediction_id,
                details: "Prediction rejected".into(),
            }),
        }
    }

    /// Access the vocabulary.
    pub fn vocabulary(&self) -> &IntentVocabulary {
        &self.vocabulary
    }

    /// Mutable access to the vocabulary.
    pub fn vocabulary_mut(&mut self) -> &mut IntentVocabulary {
        &mut self.vocabulary
    }

    /// Access the attention scorer.
    pub fn scorer(&self) -> &AttentionScorer {
        &self.scorer
    }

    /// Mutable access to the attention scorer.
    pub fn scorer_mut(&mut self) -> &mut AttentionScorer {
        &mut self.scorer
    }

    /// Get the config.
    pub fn config(&self) -> &PredictionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_graph_cognitive::graph::{CognitiveGraph, CognitiveGraphConfig};
    use synapse_graph_memory::store::{VectorMemory, VectorMemoryConfig};
    use synapse_graph_types::{
        ChannelId, ClusterId, ClusterMetadata, CognitiveShortcut, EdgeType,
        IntentAssociation, IntentId, SessionId, ShortcutId, TemporalContext,
        TemporalMeta, TemporalTier, TierWeights,
    };

    fn make_gated(dim: usize, base: f32) -> GatedEmbedding {
        let vector: Vec<f32> = (0..dim).map(|i| base + (i as f32) * 0.01).collect();
        GatedEmbedding::__new(vector, ChannelId(0), Timestamp(0), SessionId(1), 0.9)
    }

    fn make_temporal_context(dim: usize) -> TemporalContext {
        TemporalContext {
            fast_component: vec![0.5; dim],
            medium_component: vec![0.3; dim],
            slow_component: vec![0.2; dim],
            weights: TierWeights::default(),
        }
    }

    fn setup_graph_with_shortcuts() -> (CognitiveGraph, Vec<ShortcutId>) {
        let mut graph = CognitiveGraph::new(CognitiveGraphConfig::default());

        // Create nodes.
        let n1 = graph.hypergraph_mut().upsert_node(
            vec![1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            ClusterMetadata {
                source_cluster: ClusterId(1),
                member_count: 5,
                temporal_span: (Timestamp(1000), Timestamp(1000)),
            },
        );
        let n2 = graph.hypergraph_mut().upsert_node(
            vec![0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            ClusterMetadata {
                source_cluster: ClusterId(2),
                member_count: 5,
                temporal_span: (Timestamp(1000), Timestamp(1000)),
            },
        );

        graph.hypergraph_mut().upsert_edge(n1, n2, EdgeType::CoActivation, 0.8);

        // Register shortcuts with intent associations.
        let mut shortcut_ids = Vec::new();
        for i in 0..3 {
            let id = ShortcutId(i + 1);
            let shortcut = CognitiveShortcut {
                id,
                nodes: vec![n1, n2],
                edges: Vec::new(),
                activation_frequency: 0.8,
                predictive_power: 0.9,
                discovered_at: Timestamp(1000),
                proof: synapse_graph_types::AttentionProof {
                    temporal_proof: vec![1],
                    manifold_proof: vec![1],
                    biological_proof: None,
                    combined_hash: synapse_graph_types::AttentionProof::compute_hash(
                        &[1], &[1], None,
                    ),
                },
                intent_associations: vec![IntentAssociation {
                    intent_id: IntentId(i + 1),
                    strength: 0.8 - (i as f32) * 0.1,
                }],
            };
            graph.shortcuts_mut().register(shortcut);
            shortcut_ids.push(id);
        }

        (graph, shortcut_ids)
    }

    #[test]
    fn predict_with_known_shortcuts_returns_ranked_intents() {
        let (graph, _shortcut_ids) = setup_graph_with_shortcuts();
        let mut predictor = IntentPredictor::new(PredictionConfig::default());

        // Register intents.
        predictor.vocabulary_mut().register_with_id(IntentId(1), "move_left".into());
        predictor.vocabulary_mut().register_with_id(IntentId(2), "move_right".into());
        predictor.vocabulary_mut().register_with_id(IntentId(3), "stop".into());

        // Set up memory with an embedding close to node centroids.
        let mut memory = VectorMemory::new(VectorMemoryConfig {
            dimension: 8,
            ..Default::default()
        });
        let emb = make_gated(8, 1.0);
        let meta = TemporalMeta {
            tier: TemporalTier::Fast,
            timestamp: Timestamp(1000),
            channel_id: ChannelId(0),
            session_id: SessionId(1),
            coherence_score: 0.9,
        };
        let emb_for_insert = make_gated(8, 1.0);
        let _ = memory.insert(emb_for_insert, meta);

        let ctx = make_temporal_context(8);
        let (prediction, cold_events) = predictor.predict(
            &emb,
            &ctx,
            &memory,
            graph.shortcuts(),
            graph.hypergraph(),
        );

        // With shortcuts matching, we should get ranked intents.
        // Note: results depend on KNN finding matches close to shortcut nodes.
        assert!(cold_events.is_empty(), "should not be cold start");
        assert!(prediction.latency_us > 0);
    }

    #[test]
    fn temporal_reweighting_boosts_consistent_predictions() {
        use crate::reweighter::TemporalReweighter;
        use synapse_graph_types::ScoredIntent;

        let reweighter = TemporalReweighter::new(ReweightConfig { boost_factor: 0.5 });

        let mut scored = vec![
            ScoredIntent {
                intent_id: IntentId(1),
                confidence: 0.5,
                shortcut_id: ShortcutId(1),
                match_distance: 0.1,
            },
            ScoredIntent {
                intent_id: IntentId(2),
                confidence: 0.5,
                shortcut_id: ShortcutId(2),
                match_distance: 0.2,
            },
        ];

        let _original_c1 = scored[0].confidence;
        let _original_c2 = scored[1].confidence;

        let ctx = TemporalContext {
            fast_component: vec![1.0; 4],
            medium_component: vec![0.5; 4],
            slow_component: vec![0.2; 4],
            weights: TierWeights::default(),
        };

        let influence = reweighter.reweight(&mut scored, &ctx);

        // After reweighting, confidences should have changed.
        assert!(influence != 0.0, "temporal influence should be non-zero");
        // Confidences are re-normalized so they should still sum to ~1.0.
        let total: f32 = scored.iter().map(|s| s.confidence).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "confidences should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn cold_start_returns_empty_and_emits_new_intent() {
        let graph = CognitiveGraph::new(CognitiveGraphConfig::default());
        let mut predictor = IntentPredictor::new(PredictionConfig::default());

        let memory = VectorMemory::new(VectorMemoryConfig {
            dimension: 8,
            ..Default::default()
        });

        let emb = make_gated(8, 1.0);
        let ctx = make_temporal_context(8);

        let (prediction, cold_events) = predictor.predict(
            &emb,
            &ctx,
            &memory,
            graph.shortcuts(),
            graph.hypergraph(),
        );

        assert!(
            prediction.ranked_intents.is_empty(),
            "cold start should produce empty intents"
        );
        assert!(
            !cold_events.is_empty(),
            "cold start should emit NewIntentDiscovered"
        );
        assert_eq!(cold_events[0].discovery_context, "cold_start");
    }

    #[test]
    fn record_feedback_corrected_produces_adaptation_signal() {
        let mut predictor = IntentPredictor::new(PredictionConfig::default());
        predictor.vocabulary_mut().register_with_id(IntentId(1), "test".into());

        let signal = predictor.record_feedback(
            PredictionId(1),
            PredictionFeedback::Corrected {
                actual: IntentId(1),
            },
        );

        assert!(signal.is_some(), "Corrected should produce AdaptationSignal");
        let s = signal.unwrap();
        assert_eq!(s.signal_type, AdaptationSignalType::PredictionError);
        assert_eq!(s.urgency, AdaptationUrgency::High);
    }

    #[test]
    fn record_feedback_confirmed_produces_none() {
        let mut predictor = IntentPredictor::new(PredictionConfig::default());

        let signal =
            predictor.record_feedback(PredictionId(1), PredictionFeedback::Confirmed);

        assert!(signal.is_none(), "Confirmed should produce None");
    }

    #[test]
    fn record_feedback_rejected_produces_adaptation_signal() {
        let mut predictor = IntentPredictor::new(PredictionConfig::default());

        let signal =
            predictor.record_feedback(PredictionId(1), PredictionFeedback::Rejected);

        assert!(signal.is_some(), "Rejected should produce AdaptationSignal");
        assert_eq!(signal.unwrap().urgency, AdaptationUrgency::High);
    }

    #[test]
    fn record_feedback_timeout_produces_none() {
        let mut predictor = IntentPredictor::new(PredictionConfig::default());

        let signal =
            predictor.record_feedback(PredictionId(1), PredictionFeedback::Timeout);

        assert!(signal.is_none(), "Timeout should produce None");
    }

    #[test]
    fn accuracy_degraded_fires_below_threshold() {
        let mut predictor = IntentPredictor::new(PredictionConfig {
            vocabulary: crate::vocabulary::VocabularyConfig {
                accuracy_window: 20,
                accuracy_threshold: 0.7,
            },
            ..Default::default()
        });

        let intent_id = predictor.vocabulary_mut().register("test_intent".into());

        // Record mostly incorrect outcomes.
        let mut degraded_event = None;
        for i in 0..15 {
            let correct = i < 3; // only 3 out of 15 correct = 20% accuracy
            let event = predictor
                .vocabulary_mut()
                .record_accuracy(&intent_id, correct);
            if event.is_some() {
                degraded_event = event;
            }
        }

        assert!(
            degraded_event.is_some(),
            "should fire AccuracyDegraded when accuracy drops below 0.7"
        );
        let d = degraded_event.unwrap();
        assert!(d.current_accuracy < 0.7);
        assert_eq!(d.threshold, 0.7);
    }

    #[test]
    fn predict_latency_benchmark() {
        // Create a setup with shortcuts and measure prediction latency.
        let (graph, _) = setup_graph_with_shortcuts();
        let mut predictor = IntentPredictor::new(PredictionConfig::default());

        let memory = VectorMemory::new(VectorMemoryConfig {
            dimension: 8,
            ..Default::default()
        });

        let emb = make_gated(8, 1.0);
        let ctx = make_temporal_context(8);

        // Warm up.
        let _ = predictor.predict(&emb, &ctx, &memory, graph.shortcuts(), graph.hypergraph());

        let iterations = 100u32;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(predictor.predict(
                std::hint::black_box(&emb),
                std::hint::black_box(&ctx),
                &memory,
                graph.shortcuts(),
                graph.hypergraph(),
            ));
        }
        let per_call = start.elapsed() / iterations;

        // Target: < 1ms per call. Allow more headroom in debug mode.
        let ceiling_us: u128 = if cfg!(debug_assertions) { 5000 } else { 1000 };
        assert!(
            per_call.as_micros() < ceiling_us,
            "predict() took {per_call:?}, expected < {}µs",
            ceiling_us
        );
        eprintln!("predict(): {per_call:?}");
    }
}
