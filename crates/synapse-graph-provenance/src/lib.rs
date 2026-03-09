pub mod dag_store;
pub mod resolver;
pub mod query;
pub mod log;

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use synapse_graph_types::*;

    use crate::log::ProvenanceLog;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_ingested_event(id: u64) -> NeuralEvent {
        NeuralEvent::Ingested(EmbeddingIngested {
            embedding_id: EmbeddingId(id),
            gated_embedding: GatedEmbedding::__new(
                vec![0.1 * id as f32; 8],
                ChannelId(0),
                Timestamp(id * 1000),
                SessionId(1),
                0.95,
            ),
        })
    }

    fn make_stored_event(id: u64) -> NeuralEvent {
        NeuralEvent::Stored(EmbeddingStored {
            embedding_id: EmbeddingId(id),
            collection_id: None,
            metadata: TemporalMeta {
                tier: TemporalTier::Fast,
                timestamp: Timestamp(id * 1000),
                channel_id: ChannelId(0),
                session_id: SessionId(1),
                coherence_score: 0.9,
            },
            hnsw_neighbors: vec![EmbeddingId(id + 100)],
        })
    }

    fn make_promoted_event(id: u64) -> NeuralEvent {
        NeuralEvent::Promoted(PatternPromoted {
            pattern_id: PatternId(id),
            from_tier: TemporalTier::Fast,
            to_tier: TemporalTier::Medium,
            persistence_duration_us: 5_000_000,
            activation_count: 10,
        })
    }

    fn make_shortcut_found_event() -> NeuralEvent {
        let proof_temporal = vec![1u8, 2, 3, 4, 5];
        let proof_manifold = vec![6u8, 7, 8, 9, 10];
        let hash = AttentionProof::compute_hash(&proof_temporal, &proof_manifold, None);

        NeuralEvent::ShortcutFound(ShortcutDiscovered {
            shortcut: CognitiveShortcut {
                id: ShortcutId(1),
                nodes: vec![NodeId(1), NodeId(2), NodeId(3)],
                edges: Vec::new(),
                activation_frequency: 0.8,
                predictive_power: 0.9,
                discovered_at: Timestamp(10000),
                proof: AttentionProof {
                    temporal_proof: proof_temporal,
                    manifold_proof: proof_manifold,
                    biological_proof: None,
                    combined_hash: hash,
                },
                intent_associations: Vec::new(),
            },
            proof: AttentionProof {
                temporal_proof: vec![1],
                manifold_proof: vec![1],
                biological_proof: None,
                combined_hash: AttentionProof::compute_hash(&[1], &[1], None),
            },
            discovery_latency_ms: 5,
        })
    }

    fn make_predicted_event() -> NeuralEvent {
        NeuralEvent::Predicted(PredictionMade {
            prediction: IntentPrediction {
                id: PredictionId(1),
                timestamp: Timestamp(20000),
                ranked_intents: vec![ScoredIntent {
                    intent_id: IntentId(1),
                    confidence: 0.9,
                    shortcut_id: ShortcutId(1),
                    match_distance: 0.1,
                }],
                attention_weights: vec![0.9],
                temporal_influence: 0.5,
                latency_us: 500,
            },
            feedback_window_ms: 5000,
        })
    }

    fn make_adapted_event() -> NeuralEvent {
        NeuralEvent::Adapted(AdaptationApplied {
            result: AdaptationResult {
                delta: None,
                latency_us: 50,
                path: AdaptationPath::Fast,
                ewc_penalty: 0.0,
            },
            affected_layers: vec![LayerId(0), LayerId(1)],
            trigger: AdaptationType::FastCorrection,
        })
    }

    fn make_consolidated_event() -> NeuralEvent {
        NeuralEvent::Consolidated(Consolidated {
            result: ConsolidationResult {
                deltas_consolidated: 5,
                total_latency_us: 8000,
                ewc_penalty: 0.1,
                weights_protected: 10,
                fisher_recomputed: false,
            },
            accumulated_fast_deltas_count: 5,
        })
    }

    fn make_healed_event() -> NeuralEvent {
        NeuralEvent::Healed(GraphHealed {
            weak_points_reinforced: 2,
            dead_regions_pruned: 1,
            nodes_removed: 3,
            edges_added: 4,
            edges_removed: 1,
        })
    }

    fn make_topology_changed_event() -> NeuralEvent {
        NeuralEvent::TopologyChanged(GraphTopologyChanged {
            total_nodes: 50,
            total_edges: 120,
            average_min_cut: 0.5,
            shortcut_count: 5,
        })
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn append_100_mixed_events_all_retrievable() {
        let mut log = ProvenanceLog::new(SessionId(1));

        let mut ids = Vec::new();
        for i in 0..100 {
            let event = match i % 5 {
                0 => make_ingested_event(i),
                1 => make_stored_event(i),
                2 => make_promoted_event(i),
                3 => make_shortcut_found_event(),
                _ => make_predicted_event(),
            };
            ids.push(log.append(event));
        }

        assert_eq!(log.total_events(), 100);

        // Verify all are retrievable.
        for id in &ids {
            assert!(
                log.dag().get(*id).is_some(),
                "event {:?} should be retrievable",
                id
            );
        }
    }

    #[test]
    fn trace_predicted_event_causal_chain() {
        let mut log = ProvenanceLog::new(SessionId(1));

        // Build causal chain: Ingested → Stored → Promoted → ShortcutFound → Predicted.
        let _ingested_id = log.append(make_ingested_event(1));
        let _stored_id = log.append(make_stored_event(1));
        let _promoted_id = log.append(make_promoted_event(1));
        let _shortcut_id = log.append(make_shortcut_found_event());
        let predicted_id = log.append(make_predicted_event());

        let trail = log.query().trace(predicted_id);

        assert_eq!(trail.target_event, predicted_id);
        assert!(
            trail.causal_chain.len() >= 2,
            "causal chain should include at least the predicted event and its predecessor, got {:?}",
            trail.causal_chain
        );
        assert!(trail.depth >= 1, "depth should be >= 1");
        assert!(
            !trail.domains_involved.is_empty(),
            "should involve at least one domain"
        );
    }

    #[test]
    fn trace_deep_chain() {
        let mut log = ProvenanceLog::new(SessionId(1));

        // Build: Ingested → Stored → Promoted → ShortcutFound → Predicted → Adapted → Consolidated.
        log.append(make_ingested_event(1));
        log.append(make_stored_event(1));
        log.append(make_promoted_event(1));
        log.append(make_shortcut_found_event());
        log.append(make_predicted_event());
        log.append(make_adapted_event());
        let consolidated_id = log.append(make_consolidated_event());

        let trail = log.query().trace(consolidated_id);

        // Should trace back through Adapted → Predicted → ShortcutFound → ...
        assert!(
            trail.causal_chain.len() >= 3,
            "deep chain should have >= 3 events, got {}",
            trail.causal_chain.len()
        );
        assert!(
            trail.domains_involved.len() >= 2,
            "should span multiple domains"
        );
    }

    #[test]
    fn session_slice_returns_only_session_events() {
        let mut log = ProvenanceLog::new(SessionId(1));

        // Session 1 events.
        log.append(make_ingested_event(1));
        log.append(make_stored_event(2));

        // Session 2 events.
        log.set_session(SessionId(2));
        log.append(make_ingested_event(3));
        log.append(make_promoted_event(4));
        log.append(make_predicted_event());

        let query = log.query();
        let s1_events = query.session_slice(SessionId(1));
        let s2_events = query.session_slice(SessionId(2));

        assert_eq!(s1_events.len(), 2, "session 1 should have 2 events");
        assert_eq!(s2_events.len(), 3, "session 2 should have 3 events");

        for e in &s1_events {
            assert_eq!(e.session_id, SessionId(1));
        }
        for e in &s2_events {
            assert_eq!(e.session_id, SessionId(2));
        }
    }

    #[test]
    fn verify_integrity_clean_dag() {
        let mut log = ProvenanceLog::new(SessionId(1));

        for i in 0..20 {
            log.append(make_ingested_event(i));
        }

        let report = log.verify();

        assert!(report.is_valid, "clean DAG should pass integrity check");
        assert_eq!(report.total_events, 20);
        assert_eq!(report.verified_events, 20);
        assert!(report.corrupted_events.is_empty());
    }

    #[test]
    fn verify_integrity_detects_tamper() {
        let mut log = ProvenanceLog::new(SessionId(1));

        for i in 0..10 {
            log.append(make_ingested_event(i));
        }

        // Tamper with one event's payload.
        let tampered_id = EventId(5);
        if let Some(node) = log.dag_mut().get_mut(tampered_id) {
            node.payload = vec![0xFF; 10]; // corrupted payload
        }

        let report = log.verify();

        assert!(!report.is_valid, "tampered DAG should fail integrity check");
        assert!(
            !report.corrupted_events.is_empty(),
            "should detect corrupted events"
        );
        assert!(
            report.corrupted_events.contains(&tampered_id),
            "should identify the tampered event"
        );
    }

    #[test]
    fn healed_event_links_to_topology_changed() {
        let mut log = ProvenanceLog::new(SessionId(1));

        let topo_id = log.append(make_topology_changed_event());
        let healed_id = log.append(make_healed_event());

        let healed_node = log.dag().get(healed_id).unwrap();

        // Healed should reference the TopologyChanged event as predecessor.
        // TopologyChanged has EventType::SystemHealth, which Healed links to.
        assert!(
            !healed_node.predecessor_ids.is_empty(),
            "healed event should have predecessors"
        );
        // The predecessor should be the topology changed event.
        assert!(
            healed_node.predecessor_ids.contains(&topo_id),
            "healed should reference topology changed as predecessor"
        );
    }

    #[test]
    fn adapted_event_links_to_predicted() {
        let mut log = ProvenanceLog::new(SessionId(1));

        let predicted_id = log.append(make_predicted_event());
        let adapted_id = log.append(make_adapted_event());

        let adapted_node = log.dag().get(adapted_id).unwrap();

        assert!(
            adapted_node.predecessor_ids.contains(&predicted_id),
            "adapted should reference predicted as predecessor"
        );
    }

    #[test]
    fn event_types_classified_correctly() {
        assert_eq!(make_ingested_event(1).event_type(), EventType::Ingestion);
        assert_eq!(make_stored_event(1).event_type(), EventType::Storage);
        assert_eq!(make_promoted_event(1).event_type(), EventType::TierPromotion);
        assert_eq!(make_shortcut_found_event().event_type(), EventType::ShortcutDiscovery);
        assert_eq!(make_predicted_event().event_type(), EventType::Prediction);
        assert_eq!(make_adapted_event().event_type(), EventType::Adaptation);
        assert_eq!(make_consolidated_event().event_type(), EventType::Consolidation);
        assert_eq!(make_healed_event().event_type(), EventType::GraphHealing);
    }

    #[test]
    fn neural_event_serde_round_trip() {
        let events: Vec<NeuralEvent> = vec![
            make_ingested_event(1),
            make_stored_event(2),
            make_promoted_event(3),
            make_shortcut_found_event(),
            make_predicted_event(),
            make_adapted_event(),
            make_consolidated_event(),
            make_healed_event(),
            make_topology_changed_event(),
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deser: NeuralEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, &deser, "round-trip failed for {:?}", event.event_type());
        }
    }

    #[test]
    fn append_latency_benchmark() {
        let mut log = ProvenanceLog::new(SessionId(1));

        let start = Instant::now();
        for i in 0..1000 {
            log.append(make_ingested_event(i));
        }
        let elapsed = start.elapsed();
        let per_append = elapsed / 1000;

        // Target: < 5us per append (allow more in debug).
        let ceiling_us: u128 = if cfg!(debug_assertions) { 50 } else { 5 };
        assert!(
            per_append.as_micros() < ceiling_us,
            "append() took {:?} per event, expected < {}us",
            per_append, ceiling_us
        );
        eprintln!("append(): {:?} per event ({} events total)", per_append, 1000);
    }

    #[test]
    fn trace_latency_benchmark() {
        let mut log = ProvenanceLog::new(SessionId(1));

        // Build a chain of depth 10: Ingested → Stored → Promoted → ...
        for i in 0..3 {
            log.append(make_ingested_event(i));
        }
        for i in 0..3 {
            log.append(make_stored_event(i + 10));
        }
        log.append(make_promoted_event(1));
        log.append(make_shortcut_found_event());
        log.append(make_predicted_event());
        let adapted_id = log.append(make_adapted_event());

        let start = Instant::now();
        let trail = log.query().trace(adapted_id);
        let elapsed = start.elapsed();

        let ceiling_us: u128 = if cfg!(debug_assertions) { 500 } else { 100 };
        assert!(
            elapsed.as_micros() < ceiling_us,
            "trace(depth={}) took {:?}, expected < {}us",
            trail.depth, elapsed, ceiling_us
        );
        eprintln!("trace(depth={}): {:?}", trail.depth, elapsed);
    }

    #[test]
    fn adaptation_history_query() {
        let mut log = ProvenanceLog::new(SessionId(1));

        // Add adapted events for different layers.
        log.append(NeuralEvent::Adapted(AdaptationApplied {
            result: AdaptationResult {
                delta: None,
                latency_us: 50,
                path: AdaptationPath::Fast,
                ewc_penalty: 0.0,
            },
            affected_layers: vec![LayerId(0)],
            trigger: AdaptationType::FastCorrection,
        }));

        log.append(NeuralEvent::Adapted(AdaptationApplied {
            result: AdaptationResult {
                delta: None,
                latency_us: 60,
                path: AdaptationPath::Fast,
                ewc_penalty: 0.0,
            },
            affected_layers: vec![LayerId(1)],
            trigger: AdaptationType::FastCorrection,
        }));

        log.append(NeuralEvent::Adapted(AdaptationApplied {
            result: AdaptationResult {
                delta: None,
                latency_us: 70,
                path: AdaptationPath::Slow,
                ewc_penalty: 0.1,
            },
            affected_layers: vec![LayerId(0), LayerId(1)],
            trigger: AdaptationType::SlowConsolidation,
        }));

        let query = log.query();
        let layer0_history = query.adaptation_history(LayerId(0));
        let layer1_history = query.adaptation_history(LayerId(1));

        assert_eq!(layer0_history.len(), 2, "layer 0 should have 2 adaptations");
        assert_eq!(layer1_history.len(), 2, "layer 1 should have 2 adaptations");
    }

    #[test]
    fn shortcut_origin_query() {
        let mut log = ProvenanceLog::new(SessionId(1));

        log.append(make_ingested_event(1));
        let _shortcut_event_id = log.append(make_shortcut_found_event());

        let query = log.query();
        let origin = query.shortcut_origin(ShortcutId(1));
        assert!(origin.is_some(), "should find shortcut origin event");
        assert_eq!(origin.unwrap().event_type, EventType::ShortcutDiscovery);

        // Non-existent shortcut.
        let missing = query.shortcut_origin(ShortcutId(999));
        assert!(missing.is_none());
    }

    #[test]
    fn root_events_have_no_predecessors() {
        let mut log = ProvenanceLog::new(SessionId(1));

        let id = log.append(make_ingested_event(1));
        let node = log.dag().get(id).unwrap();

        assert!(
            node.predecessor_ids.is_empty(),
            "ingested event is a root event and should have no predecessors"
        );
    }

    #[test]
    fn consolidated_links_to_adapted() {
        let mut log = ProvenanceLog::new(SessionId(1));

        let adapted_id1 = log.append(make_adapted_event());
        let adapted_id2 = log.append(make_adapted_event());
        let consolidated_id = log.append(make_consolidated_event());

        let node = log.dag().get(consolidated_id).unwrap();

        assert!(
            node.predecessor_ids.contains(&adapted_id1) || node.predecessor_ids.contains(&adapted_id2),
            "consolidated should reference adapted events"
        );
    }

    #[test]
    fn integrity_report_counts() {
        let mut log = ProvenanceLog::new(SessionId(1));

        for i in 0..50 {
            log.append(make_ingested_event(i));
        }

        let report = log.verify();
        assert_eq!(report.total_events, 50);
        assert_eq!(report.verified_events, 50);
        assert!(report.is_valid);
    }
}
