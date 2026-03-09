//! OscillationDetector — detects sign-flip patterns in LoRA deltas (DDD-006).

use std::collections::HashMap;

use synapse_graph_types::{LayerId, LoraDelta};

/// Detects oscillation in LoRA deltas by tracking sign-flip patterns.
pub struct OscillationDetector {
    sign_history: HashMap<LayerId, Vec<bool>>,
    window_size: usize,
    flip_threshold: usize,
}

impl OscillationDetector {
    pub fn new(window_size: usize, flip_threshold: usize) -> Self {
        Self {
            sign_history: HashMap::new(),
            window_size,
            flip_threshold,
        }
    }

    /// Record a delta's sign direction for its layer.
    pub fn record(&mut self, layer_id: LayerId, delta: &LoraDelta) {
        // Determine sign: positive if sum of matrix_a is positive.
        let sum: f32 = delta.matrix_a.iter().sum();
        let positive = sum >= 0.0;

        let history = self.sign_history.entry(layer_id).or_default();
        history.push(positive);
        if history.len() > self.window_size {
            history.remove(0);
        }
    }

    /// Check if a layer is oscillating (sign flips exceed threshold).
    pub fn is_oscillating(&self, layer_id: &LayerId) -> bool {
        self.count_flips(layer_id) >= self.flip_threshold
    }

    /// Compute a damping multiplier for a layer.
    ///
    /// Returns 1.0 if not oscillating, lower values if oscillating.
    pub fn damping_multiplier(&self, layer_id: &LayerId) -> f32 {
        let flips = self.count_flips(layer_id);
        if flips < self.flip_threshold {
            1.0
        } else {
            // Progressively reduce learning rate as oscillation intensifies.
            let excess = flips - self.flip_threshold + 1;
            1.0 / (1.0 + excess as f32)
        }
    }

    /// Count sign flips in the history window for a layer.
    fn count_flips(&self, layer_id: &LayerId) -> usize {
        match self.sign_history.get(layer_id) {
            Some(history) if history.len() >= 2 => {
                history
                    .windows(2)
                    .filter(|w| w[0] != w[1])
                    .count()
            }
            _ => 0,
        }
    }
}
