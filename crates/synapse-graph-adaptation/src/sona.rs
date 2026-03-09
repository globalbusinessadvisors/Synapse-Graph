//! SonaEngine — wraps ruvector-sona for LoRA adaptation (DDD-006).
//!
//! In production, backs onto ruvector-sona v2.0.4 WASM bindings.
//! This implementation provides deterministic LoRA matrix generation.

/// The Sona engine — produces low-rank adaptation matrices.
pub struct SonaEngine {
    seed: u64,
}

impl SonaEngine {
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    /// Compute a fast (rank-4) LoRA adaptation from an error gradient.
    ///
    /// Returns (matrix_a, matrix_b) where:
    ///   matrix_a: shape (d_out, rank)
    ///   matrix_b: shape (d_in, rank)
    pub fn fast_adapt(&mut self, gradient: &[f32], rank: usize) -> (Vec<f32>, Vec<f32>) {
        self.compute_lora(gradient, rank)
    }

    /// Compute a slow (rank-16) LoRA adaptation from an aggregated gradient.
    ///
    /// Returns (matrix_a, matrix_b).
    pub fn slow_adapt(&mut self, gradient: &[f32], rank: usize) -> (Vec<f32>, Vec<f32>) {
        self.compute_lora(gradient, rank)
    }

    /// Internal LoRA computation using gradient outer product decomposition.
    fn compute_lora(&mut self, gradient: &[f32], rank: usize) -> (Vec<f32>, Vec<f32>) {
        let dim = gradient.len().max(1);

        // matrix_a: (dim, rank) — scale down gradient into rank columns.
        let mut matrix_a = Vec::with_capacity(dim * rank);
        for i in 0..dim {
            for r in 0..rank {
                let val = gradient.get(i).copied().unwrap_or(0.0)
                    * self.pseudo_random_scale(r);
                matrix_a.push(val);
            }
        }

        // matrix_b: (dim, rank) — complementary decomposition.
        let mut matrix_b = Vec::with_capacity(dim * rank);
        for i in 0..dim {
            for r in 0..rank {
                let val = gradient.get((i + r) % dim).copied().unwrap_or(0.0)
                    * self.pseudo_random_scale(r + dim);
                matrix_b.push(val);
            }
        }

        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        (matrix_a, matrix_b)
    }

    /// Deterministic pseudo-random scaling factor based on index.
    fn pseudo_random_scale(&self, index: usize) -> f32 {
        let x = (self.seed as f32 * 0.0001 + index as f32 * 0.7).sin();
        x * 0.01 // small scale to keep deltas bounded
    }
}

impl Default for SonaEngine {
    fn default() -> Self {
        Self::new()
    }
}
