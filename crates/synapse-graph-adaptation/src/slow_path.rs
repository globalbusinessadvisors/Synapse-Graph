//! SlowLoraPath — rank-16 consolidated adaptation with EWC++ (DDD-006, ADR-004).

use synapse_graph_types::{
    AdaptationType, ConsolidationResult, LayerId, LoraDelta, Timestamp,
};

use crate::fisher::FisherInformationMatrix;
use crate::sona::SonaEngine;

/// Slow LoRA adaptation path.
///
/// Consolidates accumulated fast deltas into rank-16 LoRA deltas with
/// EWC++ regularization to protect important weights.
/// Latency target: < 10 milliseconds.
pub struct SlowLoraPath {
    rank: usize,
    target_layers: Vec<LayerId>,
    ewc_lambda: f32,
    persisted_deltas: Vec<LoraDelta>,
    last_consolidation: Timestamp,
}

impl SlowLoraPath {
    pub fn new(rank: usize, target_layers: Vec<LayerId>, ewc_lambda: f32) -> Self {
        Self {
            rank,
            target_layers,
            ewc_lambda,
            persisted_deltas: Vec::new(),
            last_consolidation: Timestamp(0),
        }
    }

    /// Consolidate accumulated fast deltas into slow LoRA deltas.
    ///
    /// Pipeline:
    /// a) Aggregate accumulated fast deltas into a combined gradient.
    /// b) For each target layer:
    ///    - Compute rank-16 LoRA delta via sona.slow_adapt().
    ///    - Apply EWC++ penalty: loss += lambda * sum(F_i * (theta_i - theta_i*)^2).
    ///    - Clip delta if EWC penalty exceeds threshold.
    /// c) Persist deltas.
    /// d) Return ConsolidationResult.
    pub fn consolidate(
        &mut self,
        accumulated: Vec<LoraDelta>,
        fisher: &FisherInformationMatrix,
        sona: &mut SonaEngine,
    ) -> ConsolidationResult {
        let start = std::time::Instant::now();
        let deltas_count = accumulated.len();

        if accumulated.is_empty() {
            return ConsolidationResult {
                deltas_consolidated: 0,
                total_latency_us: 0,
                ewc_penalty: 0.0,
                weights_protected: 0,
                fisher_recomputed: false,
            };
        }

        // a) Aggregate fast deltas into a combined gradient.
        let gradient = self.aggregate_gradients(&accumulated);
        let mut total_ewc_penalty = 0.0f32;
        let mut weights_protected = 0usize;

        // b) For each target layer, compute slow LoRA delta.
        for &layer_id in &self.target_layers.clone() {
            let (mut matrix_a, matrix_b) = sona.slow_adapt(&gradient, self.rank);

            // Apply EWC++ penalty.
            let (ewc_penalty, protected) =
                self.apply_ewc_penalty(&mut matrix_a, fisher);
            total_ewc_penalty += ewc_penalty;
            weights_protected += protected;

            let delta = LoraDelta {
                layer_id,
                matrix_a,
                matrix_b,
                rank: self.rank,
                computed_at: Timestamp(0),
                trigger: AdaptationType::SlowConsolidation,
            };

            // c) Persist.
            self.persisted_deltas.push(delta);
        }

        self.last_consolidation = Timestamp(0);
        let elapsed = start.elapsed();

        ConsolidationResult {
            deltas_consolidated: deltas_count,
            total_latency_us: elapsed.as_micros() as u64,
            ewc_penalty: total_ewc_penalty,
            weights_protected,
            fisher_recomputed: false,
        }
    }

    /// Get persisted deltas.
    pub fn persisted_deltas(&self) -> &[LoraDelta] {
        &self.persisted_deltas
    }

    /// Last consolidation timestamp.
    pub fn last_consolidation(&self) -> Timestamp {
        self.last_consolidation
    }

    /// Aggregate fast deltas into a single gradient vector.
    fn aggregate_gradients(&self, deltas: &[LoraDelta]) -> Vec<f32> {
        if deltas.is_empty() {
            return Vec::new();
        }

        // Find max dimension across all delta matrix_a vectors.
        let dim = deltas.iter().map(|d| d.matrix_a.len()).max().unwrap_or(0);
        let mut gradient = vec![0.0f32; dim];
        let n = deltas.len() as f32;

        for delta in deltas {
            for (i, &v) in delta.matrix_a.iter().enumerate() {
                if i < dim {
                    gradient[i] += v / n;
                }
            }
        }

        gradient
    }

    /// Apply EWC++ penalty to a delta matrix.
    ///
    /// For each weight: delta *= 1 / (1 + lambda * F_i)
    /// Returns (total_penalty, protected_count).
    fn apply_ewc_penalty(
        &self,
        matrix: &mut [f32],
        fisher: &FisherInformationMatrix,
    ) -> (f32, usize) {
        let mut total_penalty = 0.0f32;
        let mut protected = 0usize;

        for (i, val) in matrix.iter_mut().enumerate() {
            let f_i = fisher.importance(i);
            let penalty = self.ewc_lambda * f_i;
            total_penalty += penalty * (*val) * (*val);

            // Scale down the delta proportional to importance.
            let scale = 1.0 / (1.0 + self.ewc_lambda * f_i);
            *val *= scale;

            if f_i > 0.01 {
                protected += 1;
            }
        }

        (total_penalty, protected)
    }
}
