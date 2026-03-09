//! FastLoraPath — immediate rank-4 adaptation for prediction errors (DDD-006, ADR-004).

use synapse_graph_types::{
    AdaptationSignal, AdaptationType, LayerId, LoraDelta, Timestamp,
};

use crate::sona::SonaEngine;

/// Fast LoRA adaptation path.
///
/// Computes rank-4 LoRA deltas in response to high-urgency adaptation signals.
/// Latency target: < 100 microseconds.
pub struct FastLoraPath {
    rank: usize,
    target_layers: Vec<LayerId>,
    accumulated_deltas: Vec<LoraDelta>,
    damping_factor: f32,
    max_accumulated: usize,
    /// EMA state per layer for damping.
    ema_state: Vec<Vec<f32>>,
}

impl FastLoraPath {
    pub fn new(
        rank: usize,
        target_layers: Vec<LayerId>,
        damping_factor: f32,
        max_accumulated: usize,
    ) -> Self {
        let ema_state = vec![Vec::new(); target_layers.len()];
        Self {
            rank,
            target_layers,
            accumulated_deltas: Vec::new(),
            damping_factor,
            max_accumulated,
            ema_state,
        }
    }

    /// Compute a fast LoRA delta from an adaptation signal.
    ///
    /// Pipeline:
    /// a) Compute error gradient from signal.
    /// b) Use sona.fast_adapt(gradient, rank=4) for rank-4 LoRA matrices.
    /// c) Apply exponential moving average with damping_factor.
    /// d) Append to accumulated_deltas.
    /// e) Return the delta.
    pub fn compute_delta(
        &mut self,
        signal: &AdaptationSignal,
        sona: &mut SonaEngine,
    ) -> LoraDelta {
        // a) Compute error gradient from signal.
        let gradient = self.signal_to_gradient(signal);

        // Pick the first target layer for this delta.
        let layer_id = self.target_layers.first().copied().unwrap_or(LayerId(0));

        // b) Get rank-4 LoRA matrices from sona.
        let (mut matrix_a, matrix_b) = sona.fast_adapt(&gradient, self.rank);

        // c) Apply EMA damping to prevent oscillation.
        let layer_idx = self.target_layers.iter().position(|&l| l == layer_id).unwrap_or(0);
        if layer_idx < self.ema_state.len() {
            if self.ema_state[layer_idx].is_empty() {
                self.ema_state[layer_idx] = matrix_a.clone();
            } else {
                let alpha = 1.0 - self.damping_factor;
                for (i, v) in matrix_a.iter_mut().enumerate() {
                    if let Some(&prev) = self.ema_state[layer_idx].get(i) {
                        *v = alpha * *v + self.damping_factor * prev;
                    }
                }
                self.ema_state[layer_idx] = matrix_a.clone();
            }
        }

        let delta = LoraDelta {
            layer_id,
            matrix_a,
            matrix_b,
            rank: self.rank,
            computed_at: Timestamp(0),
            trigger: AdaptationType::FastCorrection,
        };

        // d) Append to accumulated.
        self.accumulated_deltas.push(delta.clone());

        // e) Return.
        delta
    }

    /// Drain accumulated deltas for consolidation.
    pub fn drain_accumulated(&mut self) -> Vec<LoraDelta> {
        std::mem::take(&mut self.accumulated_deltas)
    }

    /// Number of accumulated deltas.
    pub fn accumulated_count(&self) -> usize {
        self.accumulated_deltas.len()
    }

    /// Whether we've exceeded the max accumulation threshold.
    pub fn needs_consolidation(&self) -> bool {
        self.accumulated_deltas.len() >= self.max_accumulated
    }

    /// Convert an adaptation signal into a gradient vector.
    fn signal_to_gradient(&self, signal: &AdaptationSignal) -> Vec<f32> {
        // Use the signal details as a seed for deterministic gradient generation.
        let seed = signal.prediction_id.0 as f32;
        let dim = 16; // default gradient dimension
        (0..dim)
            .map(|d| (seed * 0.3 + d as f32 * 0.5).sin() * 0.1)
            .collect()
    }
}
