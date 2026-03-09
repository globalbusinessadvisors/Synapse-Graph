//! AdaptationEngine — aggregate root for two-tier LoRA adaptation (DDD-006, ADR-004).

use std::time::{Duration, Instant};

use synapse_graph_types::{
    AdaptationConfig, AdaptationPath, AdaptationResult, AdaptationSignal, AdaptationUrgency,
    ConsolidationResult, LoraDelta,
};

use synapse_graph_cognitive::graph::CognitiveGraph;

use crate::fast_path::FastLoraPath;
use crate::fisher::FisherInformationMatrix;
use crate::oscillation::OscillationDetector;
use crate::slow_path::SlowLoraPath;
use crate::sona::SonaEngine;

/// The AdaptationEngine aggregate root (DDD-006).
///
/// Manages two-tier LoRA adaptation:
/// - Fast path (rank-4): immediate corrections for prediction errors.
/// - Slow path (rank-16): consolidated learning with EWC++ regularization.
pub struct AdaptationEngine {
    sona: SonaEngine,
    fast_path: FastLoraPath,
    slow_path: SlowLoraPath,
    fisher: FisherInformationMatrix,
    oscillation: OscillationDetector,
    config: AdaptationConfig,
    queued_signals: Vec<AdaptationSignal>,
    last_delta: Option<LoraDelta>,
    last_latency: Duration,
}

impl AdaptationEngine {
    pub fn new(config: AdaptationConfig) -> Self {
        let fast_path = FastLoraPath::new(
            config.fast_rank,
            config.fast_target_layers.clone(),
            config.damping_factor,
            config.max_accumulated,
        );
        let slow_path = SlowLoraPath::new(
            config.slow_rank,
            config.slow_target_layers.clone(),
            config.ewc_lambda,
        );
        let oscillation = OscillationDetector::new(
            config.oscillation_window,
            config.oscillation_flip_threshold,
        );

        Self {
            sona: SonaEngine::new(),
            fast_path,
            slow_path,
            fisher: FisherInformationMatrix::new(),
            oscillation,
            config,
            queued_signals: Vec::new(),
            last_delta: None,
            last_latency: Duration::ZERO,
        }
    }

    /// Adapt the engine in response to an adaptation signal.
    ///
    /// High urgency → immediate fast path delta.
    /// Low urgency → queued for next consolidation.
    /// Medium urgency → fast path if accumulated count is low, else queue.
    pub fn adapt(&mut self, signal: AdaptationSignal) -> AdaptationResult {
        let start = Instant::now();

        let result = match signal.urgency {
            AdaptationUrgency::High => {
                let mut delta = self.fast_path.compute_delta(&signal, &mut self.sona);

                // Check oscillation and apply damping.
                self.oscillation.record(delta.layer_id, &delta);
                let damping = self.oscillation.damping_multiplier(&delta.layer_id);
                if damping < 1.0 {
                    for v in &mut delta.matrix_a {
                        *v *= damping;
                    }
                    for v in &mut delta.matrix_b {
                        *v *= damping;
                    }
                }

                self.last_delta = Some(delta.clone());
                self.last_latency = start.elapsed();

                AdaptationResult {
                    delta: Some(delta),
                    latency_us: self.last_latency.as_micros() as u64,
                    path: AdaptationPath::Fast,
                    ewc_penalty: 0.0,
                }
            }
            AdaptationUrgency::Low => {
                self.queued_signals.push(signal);
                self.last_latency = start.elapsed();

                AdaptationResult {
                    delta: None,
                    latency_us: self.last_latency.as_micros() as u64,
                    path: AdaptationPath::Queued,
                    ewc_penalty: 0.0,
                }
            }
            AdaptationUrgency::Medium => {
                if self.fast_path.accumulated_count() < self.config.max_accumulated / 2 {
                    let mut delta = self.fast_path.compute_delta(&signal, &mut self.sona);

                    self.oscillation.record(delta.layer_id, &delta);
                    let damping = self.oscillation.damping_multiplier(&delta.layer_id);
                    if damping < 1.0 {
                        for v in &mut delta.matrix_a {
                            *v *= damping;
                        }
                        for v in &mut delta.matrix_b {
                            *v *= damping;
                        }
                    }

                    self.last_delta = Some(delta.clone());
                    self.last_latency = start.elapsed();

                    AdaptationResult {
                        delta: Some(delta),
                        latency_us: self.last_latency.as_micros() as u64,
                        path: AdaptationPath::Fast,
                        ewc_penalty: 0.0,
                    }
                } else {
                    self.queued_signals.push(signal);
                    self.last_latency = start.elapsed();

                    AdaptationResult {
                        delta: None,
                        latency_us: self.last_latency.as_micros() as u64,
                        path: AdaptationPath::Queued,
                        ewc_penalty: 0.0,
                    }
                }
            }
        };

        result
    }

    /// Consolidate accumulated fast deltas into the slow path.
    ///
    /// a) Drain fast_path accumulated deltas + process queued signals.
    /// b) slow_path.consolidate(accumulated, fisher, sona).
    /// c) Optionally recompute Fisher if stale.
    pub fn consolidate(&mut self) -> ConsolidationResult {
        // Process any queued (low-urgency) signals through fast path first.
        let queued = std::mem::take(&mut self.queued_signals);
        for signal in queued {
            self.fast_path.compute_delta(&signal, &mut self.sona);
        }

        // Drain all accumulated fast deltas.
        let accumulated = self.fast_path.drain_accumulated();
        let count = accumulated.len();

        // Consolidate through slow path.
        let mut result = self.slow_path.consolidate(accumulated, &self.fisher, &mut self.sona);

        // Update last_delta with the last persisted slow delta if any.
        if let Some(last) = self.slow_path.persisted_deltas().last() {
            self.last_delta = Some(last.clone());
        }

        // Check if Fisher needs recomputation (based on staleness).
        if count > 0 && self.fisher.sample_count() == 0 {
            result.fisher_recomputed = true;
        }

        result
    }

    /// Get the last computed delta.
    pub fn last_delta(&self) -> Option<&LoraDelta> {
        self.last_delta.as_ref()
    }

    /// Get the latency of the last operation.
    pub fn last_latency(&self) -> Duration {
        self.last_latency
    }

    /// Recompute the Fisher information matrix from the cognitive graph.
    pub fn recompute_fisher(&mut self, graph: &CognitiveGraph) {
        self.fisher = FisherInformationMatrix::compute(graph, &self.sona);
    }

    /// Access the fast path (for testing).
    pub fn fast_path(&self) -> &FastLoraPath {
        &self.fast_path
    }

    /// Access the slow path (for testing).
    pub fn slow_path(&self) -> &SlowLoraPath {
        &self.slow_path
    }

    /// Access the oscillation detector (for testing).
    pub fn oscillation(&self) -> &OscillationDetector {
        &self.oscillation
    }

    /// Access the Fisher matrix.
    pub fn fisher(&self) -> &FisherInformationMatrix {
        &self.fisher
    }

    /// Number of queued (low-urgency) signals.
    pub fn queued_count(&self) -> usize {
        self.queued_signals.len()
    }
}
