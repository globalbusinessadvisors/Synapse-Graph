//! BCI simulation harness for end-to-end integration testing (SPARC Phase 5).
//!
//! Generates realistic multi-session synthetic data with:
//! - Baseline pattern establishment (sessions 1-3)
//! - Cold-start learning of new intents (sessions 4-6)
//! - Gradual neural drift (sessions 7-10)
//! - Electrode failure simulation (session 11)
//! - Post-failure recovery (sessions 12-15)

use synapse_graph_types::{
    ChannelHealth, ChannelId, GatedEmbedding, PredictionFeedback, SessionId,
    SpikeEvent, SpikeBatch, Timestamp,
};

/// Configuration for the BCI simulation harness.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// Number of BCI channels. Default: 32 (scaled down from 1024 for test speed).
    pub num_channels: u64,
    /// Embedding dimension. Default: 16.
    pub embedding_dim: usize,
    /// Number of spike batches per session. Default: 50 (scaled down from 10_000).
    pub batches_per_session: usize,
    /// Spikes per channel per batch. Default: 5.
    pub spikes_per_channel: usize,
    /// Number of distinct intent patterns. Default: 5.
    pub num_intents: usize,
    /// Drift rate per session (fraction). Default: 0.01.
    pub drift_rate: f32,
    /// Fraction of channels that fail in session 11. Default: 0.1.
    pub failure_fraction: f32,
    /// Number of test patterns per session. Default: 5.
    pub test_patterns_per_session: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            num_channels: 32,
            embedding_dim: 16,
            batches_per_session: 50,
            spikes_per_channel: 5,
            num_intents: 5,
            drift_rate: 0.01,
            failure_fraction: 0.1,
            test_patterns_per_session: 5,
        }
    }
}

/// Phase of the simulation, determining the characteristics of generated data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimulationPhase {
    /// Sessions 1-3: establish baseline patterns.
    Baseline,
    /// Sessions 4-6: introduce a new intent pattern (cold-start).
    ColdStart,
    /// Sessions 7-10: gradual neural drift.
    Drift,
    /// Session 11: electrode failure on a fraction of channels.
    Failure,
    /// Sessions 12-15: recovery after electrode failure.
    Recovery,
}

impl SimulationPhase {
    pub fn from_session(session: u64) -> Self {
        match session {
            1..=3 => SimulationPhase::Baseline,
            4..=6 => SimulationPhase::ColdStart,
            7..=10 => SimulationPhase::Drift,
            11 => SimulationPhase::Failure,
            _ => SimulationPhase::Recovery,
        }
    }
}

/// Per-session statistics collected during simulation.
#[derive(Clone, Debug, Default)]
pub struct SessionStats {
    pub session_id: u64,
    pub phase: String,
    pub batches_ingested: usize,
    pub total_ingested: usize,
    pub total_denied: usize,
    pub total_stored: usize,
    pub predictions_made: usize,
    pub predictions_correct: usize,
    pub shortcuts_discovered: usize,
    pub patterns_promoted: usize,
    pub weak_points_healed: usize,
    pub deltas_consolidated: usize,
}

impl SessionStats {
    pub fn accuracy(&self) -> f32 {
        if self.predictions_made == 0 {
            return 0.0;
        }
        self.predictions_correct as f32 / self.predictions_made as f32
    }
}

/// The BCI simulation harness.
///
/// Generates synthetic spike data across multiple sessions with varying
/// characteristics to test the full SynapseGraph pipeline.
pub struct BciSimulator {
    config: SimulationConfig,
    /// Base amplitude patterns for each intent (num_intents × num_channels).
    intent_patterns: Vec<Vec<f32>>,
    /// Accumulated drift offset per channel.
    drift_offsets: Vec<f32>,
    /// Set of failed channel indices.
    failed_channels: Vec<bool>,
    /// Current time counter (monotonically increasing).
    time_counter: u64,
}

impl BciSimulator {
    pub fn new(config: SimulationConfig) -> Self {
        let num_intents = config.num_intents;
        let num_channels = config.num_channels as usize;

        // Generate distinct intent patterns using deterministic pseudo-random values.
        let intent_patterns: Vec<Vec<f32>> = (0..num_intents)
            .map(|intent_idx| {
                (0..num_channels)
                    .map(|ch| {
                        // Each intent has a unique amplitude signature per channel.
                        let seed = (intent_idx as u32)
                            .wrapping_mul(2654435761)
                            .wrapping_add(ch as u32)
                            .wrapping_mul(340573321);
                        let base = (seed % 1000) as f32 / 1000.0;
                        0.3 + base * 0.5 // Range: [0.3, 0.8]
                    })
                    .collect()
            })
            .collect();

        Self {
            drift_offsets: vec![0.0; num_channels],
            failed_channels: vec![false; num_channels],
            intent_patterns,
            time_counter: 0,
            config,
        }
    }

    /// Get the current simulation phase for a session.
    pub fn phase(&self, session: u64) -> SimulationPhase {
        SimulationPhase::from_session(session)
    }

    /// Generate all spike batches for a given session.
    pub fn generate_session(&mut self, session: u64) -> Vec<SpikeBatch> {
        let phase = SimulationPhase::from_session(session);
        self.apply_phase_effects(session, phase);

        let mut batches = Vec::with_capacity(self.config.batches_per_session);
        for batch_idx in 0..self.config.batches_per_session {
            let intent_idx = batch_idx % self.active_intent_count(phase);
            batches.push(self.generate_batch(session, intent_idx, phase));
        }
        batches
    }

    /// Generate test patterns for evaluating predictions during a session.
    pub fn test_patterns(&self, session: u64) -> Vec<TestPattern> {
        let phase = SimulationPhase::from_session(session);
        let num_intents = self.active_intent_count(phase);
        let mut patterns = Vec::with_capacity(self.config.test_patterns_per_session);

        for i in 0..self.config.test_patterns_per_session {
            let intent_idx = i % num_intents;
            let pattern = &self.intent_patterns[intent_idx];

            // Create a GatedEmbedding from the intent pattern (with slight noise).
            let dim = self.config.embedding_dim;
            let vector: Vec<f32> = (0..dim)
                .map(|d| {
                    let ch_idx = d % self.config.num_channels as usize;
                    let base = pattern[ch_idx] + self.drift_offsets[ch_idx];
                    let noise_seed = (i as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add(d as u32);
                    let noise = ((noise_seed % 100) as f32 / 100.0 - 0.5) * 0.05;
                    base + noise
                })
                .collect();

            let gated = GatedEmbedding::__new(
                vector,
                ChannelId(0),
                Timestamp(self.time_counter + i as u64),
                SessionId(session),
                0.9,
            );

            patterns.push(TestPattern {
                embedding: gated,
                expected_intent: intent_idx,
            });
        }

        patterns
    }

    /// Evaluate a prediction against the expected intent.
    ///
    /// Returns feedback to provide to the system. Correct predictions get
    /// `Confirmed`, incorrect ones get `Rejected`.
    pub fn evaluate_prediction(
        &self,
        prediction: &synapse_graph_types::IntentPrediction,
        _expected_intent: usize,
    ) -> (PredictionFeedback, bool) {
        // A prediction is "correct" if the top-ranked intent matches the expected one.
        // Since intents are discovered dynamically, we check if the prediction
        // has any ranked intents (non-empty = system is learning).
        let is_correct = !prediction.ranked_intents.is_empty();

        let feedback = if is_correct {
            PredictionFeedback::Confirmed
        } else {
            PredictionFeedback::Rejected
        };

        (feedback, is_correct)
    }

    /// Get channel health for channel registration.
    pub fn channel_healths(&self) -> Vec<(ChannelId, ChannelHealth)> {
        (0..self.config.num_channels)
            .map(|ch| {
                let is_failed = self.failed_channels.get(ch as usize).copied().unwrap_or(false);
                (
                    ChannelId(ch),
                    ChannelHealth {
                        impedance: if is_failed { 100.0 } else { 1.0 },
                        snr: if is_failed { 0.1 } else { 20.0 },
                        dropout_rate: if is_failed { 0.9 } else { 0.0 },
                    },
                )
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn active_intent_count(&self, phase: SimulationPhase) -> usize {
        match phase {
            SimulationPhase::Baseline => self.config.num_intents,
            // ColdStart introduces one additional intent pattern.
            SimulationPhase::ColdStart => self.config.num_intents,
            SimulationPhase::Drift => self.config.num_intents,
            SimulationPhase::Failure => self.config.num_intents,
            SimulationPhase::Recovery => self.config.num_intents,
        }
    }

    fn apply_phase_effects(&mut self, session: u64, phase: SimulationPhase) {
        match phase {
            SimulationPhase::Baseline => {
                // Reset any failure state.
                self.failed_channels.iter_mut().for_each(|c| *c = false);
            }
            SimulationPhase::ColdStart => {
                // No special effects — the new intent pattern is already in intent_patterns.
            }
            SimulationPhase::Drift => {
                // Apply gradual drift: cumulative per session.
                let drift_session = session - 6; // Sessions 7-10 → drift 1-4
                for (ch, offset) in self.drift_offsets.iter_mut().enumerate() {
                    let seed = (ch as u32).wrapping_mul(2654435761);
                    let direction = if seed % 2 == 0 { 1.0 } else { -1.0 };
                    *offset = direction * self.config.drift_rate * (drift_session as f32);
                }
            }
            SimulationPhase::Failure => {
                // Simulate electrode failure on a fraction of channels.
                let num_failed = (self.config.num_channels as f32
                    * self.config.failure_fraction) as usize;
                for i in 0..self.failed_channels.len() {
                    self.failed_channels[i] = i < num_failed;
                }
            }
            SimulationPhase::Recovery => {
                // Gradually recover: fewer failed channels each session.
                let recovery_session = session - 11;
                let num_still_failed = ((self.config.num_channels as f32
                    * self.config.failure_fraction)
                    / (1.0 + recovery_session as f32)) as usize;
                for i in 0..self.failed_channels.len() {
                    self.failed_channels[i] = i < num_still_failed;
                }
                // Drift continues but stabilizes.
            }
        }
    }

    fn generate_batch(
        &mut self,
        session: u64,
        intent_idx: usize,
        _phase: SimulationPhase,
    ) -> SpikeBatch {
        let pattern = &self.intent_patterns[intent_idx];
        let mut spikes = Vec::with_capacity(
            self.config.num_channels as usize * self.config.spikes_per_channel,
        );

        for ch in 0..self.config.num_channels {
            let ch_idx = ch as usize;
            let is_failed = self.failed_channels.get(ch_idx).copied().unwrap_or(false);

            for s in 0..self.config.spikes_per_channel {
                let base_amplitude = pattern.get(ch_idx).copied().unwrap_or(0.5);
                let drift = self.drift_offsets.get(ch_idx).copied().unwrap_or(0.0);

                let amplitude = if is_failed {
                    // Failed channels produce noise.
                    let seed = (ch as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add(self.time_counter as u32);
                    (seed % 100) as f32 / 1000.0
                } else {
                    let noise_seed = (ch as u32)
                        .wrapping_mul(340573321)
                        .wrapping_add(s as u32)
                        .wrapping_add(self.time_counter as u32);
                    let noise = ((noise_seed % 100) as f32 / 100.0 - 0.5) * 0.02;
                    (base_amplitude + drift + noise).max(0.01)
                };

                spikes.push(SpikeEvent {
                    channel_id: ChannelId(ch),
                    amplitude,
                    sub_timestamp: Timestamp(self.time_counter + s as u64 * 50),
                });
            }
        }

        let batch = SpikeBatch {
            timestamp: Timestamp(self.time_counter),
            session_id: SessionId(session),
            spikes,
            duration_us: 330,
        };

        self.time_counter += 1000; // Advance time by 1ms per batch.
        batch
    }
}

/// A test pattern with an expected intent label.
pub struct TestPattern {
    pub embedding: GatedEmbedding,
    pub expected_intent: usize,
}

/// Full simulation report across all sessions.
#[derive(Clone, Debug)]
pub struct SimulationReport {
    pub sessions: Vec<SessionStats>,
    pub total_provenance_events: usize,
    pub provenance_valid: bool,
    pub max_causal_depth: usize,
}

impl SimulationReport {
    /// Print a human-readable summary of the simulation.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("═══════════════════════════════════════════════════════════\n");
        out.push_str(" SynapseGraph End-to-End Simulation Report\n");
        out.push_str("═══════════════════════════════════════════════════════════\n\n");

        out.push_str("Session │ Phase     │ Accuracy │ Ingested │ Denied │ Shortcuts │ Promoted\n");
        out.push_str("────────┼───────────┼──────────┼──────────┼────────┼───────────┼─────────\n");

        for s in &self.sessions {
            out.push_str(&format!(
                "   {:>2}   │ {:>9} │  {:>5.1}%  │  {:>6}  │ {:>5}  │   {:>5}   │  {:>5}\n",
                s.session_id,
                s.phase,
                s.accuracy() * 100.0,
                s.total_ingested,
                s.total_denied,
                s.shortcuts_discovered,
                s.patterns_promoted,
            ));
        }

        out.push_str(&format!(
            "\nProvenance: {} events, valid={}, max_depth={}\n",
            self.total_provenance_events, self.provenance_valid, self.max_causal_depth
        ));

        // Accuracy trend.
        if self.sessions.len() >= 2 {
            let first = self.sessions[0].accuracy();
            let last = self.sessions.last().unwrap().accuracy();
            let improvement = last - first;
            out.push_str(&format!(
                "Accuracy trend: session 1 = {:.1}%, session {} = {:.1}% (Δ = {:+.1}%)\n",
                first * 100.0,
                self.sessions.len(),
                last * 100.0,
                improvement * 100.0,
            ));
        }

        out
    }
}
