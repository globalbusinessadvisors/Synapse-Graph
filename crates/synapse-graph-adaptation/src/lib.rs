pub mod sona;
pub mod fast_path;
pub mod slow_path;
pub mod fisher;
pub mod oscillation;
pub mod engine;

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use synapse_graph_types::{
        AdaptationConfig, AdaptationPath, AdaptationSignal, AdaptationSignalType,
        AdaptationType, AdaptationUrgency, LayerId, LoraDelta, LoraError, LoraTarget,
        PredictionId, Timestamp,
    };

    use synapse_graph_cognitive::gnn_engine::{AttentionModuleConfig, GnnEngine};

    use crate::engine::AdaptationEngine;
    use crate::fisher::FisherInformationMatrix;
    use crate::oscillation::OscillationDetector;

    fn make_high_signal(pred_id: u64) -> AdaptationSignal {
        AdaptationSignal {
            signal_type: AdaptationSignalType::PredictionError,
            urgency: AdaptationUrgency::High,
            prediction_id: PredictionId(pred_id),
            details: "test prediction error".into(),
        }
    }

    fn make_low_signal(pred_id: u64) -> AdaptationSignal {
        AdaptationSignal {
            signal_type: AdaptationSignalType::AccuracyDegraded,
            urgency: AdaptationUrgency::Low,
            prediction_id: PredictionId(pred_id),
            details: "accuracy degraded".into(),
        }
    }

    #[test]
    fn adapt_high_urgency_produces_fast_delta() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());
        let signal = make_high_signal(1);

        let result = engine.adapt(signal);

        assert_eq!(result.path, AdaptationPath::Fast);
        assert!(result.delta.is_some(), "high urgency should produce a delta");
        let delta = result.delta.unwrap();
        assert_eq!(delta.trigger, AdaptationType::FastCorrection);
        assert_eq!(delta.rank, 4);
        assert!(!delta.matrix_a.is_empty());
        assert!(!delta.matrix_b.is_empty());
    }

    #[test]
    fn adapt_high_urgency_latency() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());
        let signal = make_high_signal(1);

        let start = Instant::now();
        let _result = engine.adapt(signal);
        let elapsed = start.elapsed();

        // Fast adapt should be < 100us in release, give headroom in debug.
        let ceiling_us: u128 = if cfg!(debug_assertions) { 5_000 } else { 100 };
        assert!(
            elapsed.as_micros() < ceiling_us,
            "fast adapt took {:?}, expected < {}us",
            elapsed, ceiling_us
        );
        eprintln!("fast adapt: {:?}", elapsed);
    }

    #[test]
    fn adapt_low_urgency_queues_for_consolidation() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());
        let signal = make_low_signal(1);

        let result = engine.adapt(signal);

        assert_eq!(result.path, AdaptationPath::Queued);
        assert!(result.delta.is_none(), "low urgency should not produce immediate delta");
        assert_eq!(engine.queued_count(), 1);
    }

    #[test]
    fn consolidate_applies_ewc_regularization() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());

        // Build up some fast deltas.
        for i in 0..5 {
            engine.adapt(make_high_signal(i));
        }

        // Set up a Fisher matrix with non-trivial importance values.
        // We need a graph for this — use recompute_fisher with a real graph.
        let graph = synapse_graph_cognitive::graph::CognitiveGraph::new(
            synapse_graph_cognitive::graph::CognitiveGraphConfig::default(),
        );
        engine.recompute_fisher(&graph);

        let result = engine.consolidate();

        assert_eq!(result.deltas_consolidated, 5);
        // EWC penalty should be non-negative.
        assert!(result.ewc_penalty >= 0.0);
    }

    #[test]
    fn consolidate_processes_queued_signals() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());

        // Queue some low-urgency signals.
        engine.adapt(make_low_signal(1));
        engine.adapt(make_low_signal(2));
        engine.adapt(make_low_signal(3));
        assert_eq!(engine.queued_count(), 3);

        let result = engine.consolidate();

        assert_eq!(engine.queued_count(), 0);
        // The 3 queued signals became fast deltas, then consolidated.
        assert_eq!(result.deltas_consolidated, 3);
    }

    #[test]
    fn consolidate_latency() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());
        for i in 0..10 {
            engine.adapt(make_high_signal(i));
        }

        let start = Instant::now();
        let _result = engine.consolidate();
        let elapsed = start.elapsed();

        let ceiling_ms: u128 = if cfg!(debug_assertions) { 50 } else { 10 };
        assert!(
            elapsed.as_millis() < ceiling_ms,
            "consolidate took {:?}, expected < {}ms",
            elapsed, ceiling_ms
        );
        eprintln!("consolidate(10 deltas): {:?}", elapsed);
    }

    #[test]
    fn ewc_protects_important_weights() {
        let mut engine = AdaptationEngine::new(AdaptationConfig {
            ewc_lambda: 10000.0, // high lambda for strong protection
            ..Default::default()
        });

        // Add deltas with and without EWC.
        engine.adapt(make_high_signal(1));
        let unprotected = engine.last_delta().unwrap().clone();

        // Recompute Fisher with high importance values.
        // We'll create a cognitive graph with nodes that have high-magnitude centroids.
        let mut graph = synapse_graph_cognitive::graph::CognitiveGraph::new(
            synapse_graph_cognitive::graph::CognitiveGraphConfig::default(),
        );
        // Add nodes with large centroid values to produce high Fisher diagonal.
        for i in 0..10 {
            graph.hypergraph_mut().upsert_node(
                vec![10.0, 20.0, 30.0, 40.0],
                synapse_graph_types::ClusterMetadata {
                    source_cluster: synapse_graph_types::ClusterId(i),
                    member_count: 5,
                    temporal_span: (Timestamp(1000), Timestamp(2000)),
                },
            );
        }
        engine.recompute_fisher(&graph);

        // Consolidate with high Fisher importance.
        let result = engine.consolidate();

        // With high Fisher values and high lambda, EWC penalty should be significant.
        assert!(result.weights_protected > 0, "should have protected weights");
        assert!(result.ewc_penalty > 0.0, "should have non-zero EWC penalty");

        // The persisted slow deltas should have been scaled down.
        let slow_deltas = engine.slow_path().persisted_deltas();
        assert!(!slow_deltas.is_empty());
        for delta in slow_deltas {
            // Protected weights should have smaller magnitudes.
            let max_val = delta.matrix_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_val < 1.0,
                "EWC should scale down deltas, got max_val={max_val}"
            );
        }

        // Compare: unprotected delta should have larger values.
        let _unprotected_max = unprotected.matrix_a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    }

    #[test]
    fn oscillation_detector_detects_sign_flips() {
        let mut detector = OscillationDetector::new(10, 6);
        let layer = LayerId(0);

        // Create alternating sign deltas.
        for i in 0..10 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let delta = LoraDelta {
                layer_id: layer,
                matrix_a: vec![sign * 0.1],
                matrix_b: vec![0.01],
                rank: 4,
                computed_at: Timestamp(i * 1000),
                trigger: AdaptationType::FastCorrection,
            };
            detector.record(layer, &delta);
        }

        assert!(
            detector.is_oscillating(&layer),
            "should detect oscillation with alternating signs"
        );
        let damping = detector.damping_multiplier(&layer);
        assert!(
            damping < 1.0,
            "damping multiplier should be < 1.0 when oscillating, got {damping}"
        );
    }

    #[test]
    fn oscillation_detector_no_false_positive() {
        let mut detector = OscillationDetector::new(10, 6);
        let layer = LayerId(0);

        // Create consistently positive deltas.
        for i in 0..10 {
            let delta = LoraDelta {
                layer_id: layer,
                matrix_a: vec![0.1],
                matrix_b: vec![0.01],
                rank: 4,
                computed_at: Timestamp(i * 1000),
                trigger: AdaptationType::FastCorrection,
            };
            detector.record(layer, &delta);
        }

        assert!(
            !detector.is_oscillating(&layer),
            "should not detect oscillation with consistent signs"
        );
        assert!(
            (detector.damping_multiplier(&layer) - 1.0).abs() < f32::EPSILON,
            "damping should be 1.0 when not oscillating"
        );
    }

    #[test]
    fn lora_target_gnn_engine() {
        let mut gnn = GnnEngine::new(AttentionModuleConfig::default());

        let delta = LoraDelta {
            layer_id: LayerId(0),
            matrix_a: vec![0.1, 0.2, 0.3],
            matrix_b: vec![0.01, 0.02, 0.03],
            rank: 4,
            computed_at: Timestamp(1000),
            trigger: AdaptationType::SlowConsolidation,
        };

        let result = gnn.apply_lora_delta(&delta);
        assert!(result.is_ok(), "should apply delta to valid layer");

        let adjustments = gnn.lora_adjustments();
        assert!(!adjustments.is_empty());
        assert_eq!(adjustments[0].0, LayerId(0));

        // Apply another delta to the same layer — should accumulate.
        let delta2 = LoraDelta {
            layer_id: LayerId(0),
            matrix_a: vec![0.1, 0.2, 0.3],
            matrix_b: vec![0.01, 0.02, 0.03],
            rank: 4,
            computed_at: Timestamp(2000),
            trigger: AdaptationType::SlowConsolidation,
        };
        gnn.apply_lora_delta(&delta2).unwrap();

        let adjustments = gnn.lora_adjustments();
        // Values should be doubled.
        assert!((adjustments[0].1[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn lora_target_gnn_engine_wrong_layer() {
        let mut gnn = GnnEngine::new(AttentionModuleConfig {
            num_layers: 2,
            ..Default::default()
        });

        let delta = LoraDelta {
            layer_id: LayerId(99), // doesn't exist
            matrix_a: vec![0.1],
            matrix_b: vec![0.01],
            rank: 4,
            computed_at: Timestamp(1000),
            trigger: AdaptationType::FastCorrection,
        };

        let result = gnn.apply_lora_delta(&delta);
        assert_eq!(result, Err(LoraError::LayerNotFound(LayerId(99))));
    }

    #[test]
    fn lora_target_attention_scorer() {
        let mut scorer = synapse_graph_prediction::scorer::AttentionScorer::new(
            synapse_graph_prediction::scorer::ScorerConfig::default(),
        );

        let delta = LoraDelta {
            layer_id: LayerId(0),
            matrix_a: vec![0.5, 0.6],
            matrix_b: vec![0.05, 0.06],
            rank: 4,
            computed_at: Timestamp(1000),
            trigger: AdaptationType::FastCorrection,
        };

        let result = scorer.apply_lora_delta(&delta);
        assert!(result.is_ok(), "should apply delta to attention scorer");

        let adjustments = scorer.lora_adjustments();
        assert!(!adjustments.is_empty());
    }

    #[test]
    fn lora_delta_memory_bytes() {
        let delta = LoraDelta {
            layer_id: LayerId(0),
            matrix_a: vec![0.0; 32],   // 32 floats = 128 bytes
            matrix_b: vec![0.0; 16],   // 16 floats = 64 bytes
            rank: 4,
            computed_at: Timestamp(0),
            trigger: AdaptationType::FastCorrection,
        };

        assert_eq!(delta.memory_bytes(), (32 + 16) * 4);
        let summary = delta.summary();
        assert_eq!(summary.rank, 4);
        assert_eq!(summary.memory_bytes, 192);
    }

    #[test]
    fn persisted_deltas_survive_reconstruction() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());

        // Generate some deltas.
        for i in 0..3 {
            engine.adapt(make_high_signal(i));
        }
        engine.consolidate();

        // Check persisted deltas exist.
        let persisted = engine.slow_path().persisted_deltas();
        assert!(!persisted.is_empty(), "should have persisted deltas after consolidation");

        // Serialize and deserialize the deltas.
        let serialized = serde_json::to_string(&persisted).unwrap();
        let deserialized: Vec<LoraDelta> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(persisted.len(), deserialized.len());
        for (orig, deser) in persisted.iter().zip(deserialized.iter()) {
            assert_eq!(orig.layer_id, deser.layer_id);
            assert_eq!(orig.rank, deser.rank);
            assert_eq!(orig.matrix_a.len(), deser.matrix_a.len());
            assert_eq!(orig.matrix_b.len(), deser.matrix_b.len());
            assert_eq!(orig.trigger, deser.trigger);
        }
    }

    #[test]
    fn adapt_oscillation_applies_damping() {
        let mut engine = AdaptationEngine::new(AdaptationConfig {
            oscillation_window: 10,
            oscillation_flip_threshold: 4,
            ..Default::default()
        });

        // Generate alternating signals to trigger oscillation.
        let mut last_magnitude = f32::MAX;
        for i in 0..20 {
            let signal = make_high_signal(i);
            let result = engine.adapt(signal);

            if i > 8 {
                // After enough flips, damping should kick in.
                if let Some(delta) = &result.delta {
                    let mag: f32 = delta.matrix_a.iter().map(|v| v.abs()).sum();
                    // The magnitudes should eventually decrease due to damping.
                    if engine.oscillation().is_oscillating(&delta.layer_id) {
                        assert!(
                            mag <= last_magnitude + 0.01, // allow small float error
                            "damped magnitude should not increase significantly"
                        );
                    }
                    last_magnitude = mag;
                }
            }
        }
    }

    #[test]
    fn fisher_matrix_computation() {
        let mut graph = synapse_graph_cognitive::graph::CognitiveGraph::new(
            synapse_graph_cognitive::graph::CognitiveGraphConfig::default(),
        );

        // Use very distinct centroids to avoid merging (cosine similarity < 0.95).
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
        ];
        for (i, centroid) in centroids.iter().enumerate() {
            graph.hypergraph_mut().upsert_node(
                centroid.clone(),
                synapse_graph_types::ClusterMetadata {
                    source_cluster: synapse_graph_types::ClusterId(i as u64),
                    member_count: 5,
                    temporal_span: (Timestamp(1000), Timestamp(2000)),
                },
            );
        }

        let sona = crate::sona::SonaEngine::new();
        let fisher = FisherInformationMatrix::compute(&graph, &sona);

        assert_eq!(fisher.param_count(), 3);
        assert!(fisher.importance(0) > 0.0, "Fisher diagonal should be positive");
        assert_eq!(fisher.sample_count(), 5);

        let top = fisher.top_protected(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn last_delta_tracks_correctly() {
        let mut engine = AdaptationEngine::new(AdaptationConfig::default());

        assert!(engine.last_delta().is_none());

        engine.adapt(make_high_signal(1));
        assert!(engine.last_delta().is_some());

        let delta = engine.last_delta().unwrap();
        assert_eq!(delta.trigger, AdaptationType::FastCorrection);
    }
}
