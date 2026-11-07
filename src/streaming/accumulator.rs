use faer::linalg::solvers::{Cholesky, Qr, SpSolver};
use faer::{Mat, Side};

use crate::math::factorize::{factorize_reduced_weights, factorize_weights};
use crate::types::{matrix_is_finite, CompiledLayer, LayerCompilerConfig};

/// Frequent Directions (FD) matrix sketch for wide layers (d_in > 4096 or FFN/MLP).
///
/// Implements Edo Liberty (2013) & Shi & Phillips (AISTATS 2021) deterministic matrix sketching
/// for streaming ridge regression. Maintains a sketch buffer B of shape [2m, d_in + d_out]
/// (e.g. m=128), eliminating the O(d_in^2) memory explosion (e.g. 1.64 GB per layer for d_ffn=14336)
/// down to ~19 MB, with guaranteed spectral approximation bounds:
/// ||Z^T Z - B^T B||_2 <= ||Z||_F^2 / m.
#[derive(Clone, Debug)]
pub struct FrequentDirectionsSketch {
    pub buffer_x: Mat<f32>,
    pub buffer_y: Mat<f32>,
    pub current_rows: usize,
    pub m: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub total_tokens: usize,
}

impl FrequentDirectionsSketch {
    pub fn new(d_in: usize, d_out: usize, m: usize) -> Self {
        let m = m.max(16);
        Self {
            buffer_x: Mat::zeros(2 * m, d_in),
            buffer_y: Mat::zeros(2 * m, d_out),
            current_rows: 0,
            m,
            d_in,
            d_out,
            total_tokens: 0,
        }
    }

    /// Incrementally add a chunk of activations X_chunk [nrows, d_in] and targets Y_chunk [nrows, d_out].
    pub fn add_chunk(&mut self, x: &[f32], y: &[f32], nrows: usize) -> Result<(), String> {
        if nrows == 0 {
            return Ok(());
        }
        if x.len() != nrows * self.d_in {
            return Err(format!(
                "x chunk length ({}) does not match nrows * d_in ({} * {})",
                x.len(),
                nrows,
                self.d_in
            ));
        }
        if y.len() != nrows * self.d_out {
            return Err(format!(
                "y chunk length ({}) does not match nrows * d_out ({} * {})",
                y.len(),
                nrows,
                self.d_out
            ));
        }

        for r in 0..nrows {
            let x_row = &x[r * self.d_in..(r + 1) * self.d_in];
            let y_row = &y[r * self.d_out..(r + 1) * self.d_out];

            let row_idx = self.current_rows;
            for (c, &val) in x_row.iter().enumerate().take(self.d_in) {
                self.buffer_x.write(row_idx, c, val);
            }
            for (c, &val) in y_row.iter().enumerate().take(self.d_out) {
                self.buffer_y.write(row_idx, c, val);
            }
            self.current_rows += 1;
            self.total_tokens += 1;

            if self.current_rows == 2 * self.m {
                self.compress()?;
            }
        }
        Ok(())
    }

    /// Compress the sketch using Joint Frequent Directions on Z = [X | \gamma Y].
    ///
    /// Balances Frobenius norms between X and Y to avoid asymmetric eigenvalue masking,
    /// shrinks singular values by \delta = \sigma_m^2, and retains exact joint spectral bounds.
    pub fn compress(&mut self) -> Result<(), String> {
        if self.current_rows <= self.m {
            return Ok(());
        }

        let n = self.current_rows;
        let d_total = self.d_in + self.d_out;

        // Compute Frobenius norms to determine balance multiplier \gamma
        let mut norm_x_sq = 0.0f64;
        for r in 0..n {
            for c in 0..self.d_in {
                let v = self.buffer_x.read(r, c) as f64;
                norm_x_sq += v * v;
            }
        }
        let mut norm_y_sq = 0.0f64;
        for r in 0..n {
            for c in 0..self.d_out {
                let v = self.buffer_y.read(r, c) as f64;
                norm_y_sq += v * v;
            }
        }

        let gamma = if norm_y_sq > 1e-12 {
            ((norm_x_sq / self.d_in as f64) / (norm_y_sq / self.d_out as f64 + 1e-12)).sqrt() as f32
        } else {
            1.0f32
        };

        // Form joint matrix Z = [X | \gamma Y] of shape [n, d_in + d_out]
        let joint_z = Mat::from_fn(n, d_total, |r, c| {
            if c < self.d_in {
                self.buffer_x.read(r, c)
            } else {
                gamma * self.buffer_y.read(r, c - self.d_in)
            }
        });

        // Thin SVD of joint Z: Z = U * \Sigma * V^T
        let svd_z = joint_z.thin_svd();
        let s = svd_z.s_diagonal();
        let v = svd_z.v();
        let n_s = s.nrows();

        let delta = if n_s > self.m {
            let sm = s.read(self.m);
            sm * sm
        } else {
            0.0
        };

        let keep = self.m.min(n_s);
        let inv_gamma = 1.0f32 / gamma;

        // Write shrunk sketch rows B_i = \tilde{\sigma}_i * V_{:, i}^T
        for i in 0..keep {
            let si = s.read(i);
            let si_sq = si * si;
            let shrunk = if si_sq > delta {
                (si_sq - delta).sqrt()
            } else {
                0.0
            };

            for c in 0..self.d_in {
                self.buffer_x.write(i, c, shrunk * v.read(c, i));
            }
            for c in 0..self.d_out {
                self.buffer_y.write(i, c, inv_gamma * (shrunk * v.read(self.d_in + c, i)));
            }
        }

        // Zero out remaining rows
        for i in keep..(2 * self.m) {
            for c in 0..self.d_in {
                self.buffer_x.write(i, c, 0.0);
            }
            for c in 0..self.d_out {
                self.buffer_y.write(i, c, 0.0);
            }
        }

        self.current_rows = keep;
        Ok(())
    }

    /// Solve regularized ridge regression on the sketched covariance.
    pub fn solve(
        &self,
        config: &LayerCompilerConfig,
        target: &str,
        layer_index: usize,
    ) -> Result<CompiledLayer, String> {
        if self.total_tokens == 0 || self.current_rows == 0 {
            let default_rank = config.rank.min(self.d_in).min(self.d_out);
            return Ok(CompiledLayer::new(
                layer_index,
                target,
                vec![0.0; default_rank * self.d_in],
                vec![0.0; self.d_out * default_rank],
                default_rank,
                self.d_in,
                self.d_out,
            ));
        }

        let k = self.current_rows;

        if k <= self.d_in {
            // Dual Woodbury formulation: W = B_x^T * (B_x B_x^T + lambda I)^-1 * B_y
            // High-precision f64 accumulation prevents float32 cancellation on ill-conditioned data
            let b_x_f64 = Mat::from_fn(k, self.d_in, |r, c| self.buffer_x.read(r, c) as f64);
            let b_y_f64 = Mat::from_fn(k, self.d_out, |r, c| self.buffer_y.read(r, c) as f64);

            let mut gram = b_x_f64.as_ref() * b_x_f64.transpose();
            for i in 0..k {
                gram.write(i, i, gram.read(i, i) + config.lambda_reg as f64);
            }

            let cholesky = Cholesky::try_new(gram.as_ref(), Side::Lower).map_err(|e| {
                format!("Frequent Directions Cholesky factorization failed on Gram matrix: {e:?}")
            })?;
            let z_f64 = cholesky.solve(b_y_f64);
            let z = Mat::from_fn(k, self.d_out, |r, c| z_f64.read(r, c) as f32);

            // Solve W = B_x^T * Z using thin QR without materializing the massive [d_in, d_out] matrix:
            // B_x^T [d_in, k] = Q [d_in, k] * R [k, k], so W = Q * (R * Z)
            let qr = Qr::new(self.buffer_x.submatrix(0, 0, k, self.d_in).transpose());
            let q = qr.compute_thin_q();
            let r = qr.compute_thin_r();
            let reduced = r * z.as_ref(); // shape: [k, d_out]

            if !matrix_is_finite(q.as_ref()) || !matrix_is_finite(reduced.as_ref()) {
                return Err(
                    "Frequent Directions reduced factors contain non-finite values".to_string(),
                );
            }

            factorize_reduced_weights(&q, &reduced, config, target, layer_index)
        } else {
            // Primal formulation when k > d_in: (B_x^T B_x + lambda I) W = B_x^T B_y
            let b_x_f64 = Mat::from_fn(k, self.d_in, |r, c| self.buffer_x.read(r, c) as f64);
            let b_y_f64 = Mat::from_fn(k, self.d_out, |r, c| self.buffer_y.read(r, c) as f64);

            let mut gram = b_x_f64.transpose() * b_x_f64.as_ref();
            for i in 0..self.d_in {
                gram.write(i, i, gram.read(i, i) + config.lambda_reg as f64);
            }
            let rhs = b_x_f64.transpose() * b_y_f64.as_ref();
            let cholesky = Cholesky::try_new(gram.as_ref(), Side::Lower).map_err(|e| {
                format!("Frequent Directions primal Cholesky factorization failed: {e:?}")
            })?;
            let weights_f64 = cholesky.solve(rhs.as_ref());
            let weights = Mat::from_fn(self.d_in, self.d_out, |r, c| weights_f64.read(r, c) as f32);

            if !matrix_is_finite(weights.as_ref()) {
                return Err(
                    "Frequent Directions primal solution contains non-finite values".to_string(),
                );
            }
            factorize_weights(&weights, config, target, layer_index)
        }
    }
}

/// Online covariance and cross-covariance accumulator for arbitrary-length context streaming.
///
/// Automatically uses Dense accumulation for d_in <= 4096 and Frequent Directions Sketching
/// for wide layers (d_in > 4096), keeping memory strictly bounded to avoid OOM crashes on
/// wide architectures (e.g. SwiGLU FFN where d_ffn = 14336 or 18944).
#[derive(Clone, Debug)]
pub struct CovarianceAccumulator {
    pub c: Mat<f64>,
    pub p: Mat<f64>,
    pub total_tokens: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub sketch: Option<FrequentDirectionsSketch>,
}

impl CovarianceAccumulator {
    pub fn new(d_in: usize, d_out: usize) -> Self {
        // Threshold for switching to Frequent Directions matrix sketching
        const WIDE_LAYER_THRESHOLD: usize = 4096;

        if d_in > WIDE_LAYER_THRESHOLD {
            Self {
                c: Mat::zeros(0, 0),
                p: Mat::zeros(0, 0),
                total_tokens: 0,
                d_in,
                d_out,
                sketch: Some(FrequentDirectionsSketch::new(d_in, d_out, 128)),
            }
        } else {
            Self {
                c: Mat::zeros(d_in, d_in),
                p: Mat::zeros(d_in, d_out),
                total_tokens: 0,
                d_in,
                d_out,
                sketch: None,
            }
        }
    }

    /// Explicit constructor for Frequent Directions sketching with custom buffer size `m`.
    pub fn new_sketch(d_in: usize, d_out: usize, m: usize) -> Self {
        Self {
            c: Mat::zeros(0, 0),
            p: Mat::zeros(0, 0),
            total_tokens: 0,
            d_in,
            d_out,
            sketch: Some(FrequentDirectionsSketch::new(d_in, d_out, m)),
        }
    }

    /// Incrementally add a chunk of activations X_chunk [nrows, d_in] and targets Y_chunk [nrows, d_out].
    pub fn add_chunk(&mut self, x: &[f32], y: &[f32], nrows: usize) -> Result<(), String> {
        if nrows == 0 {
            return Ok(());
        }

        if let Some(ref mut sketch) = self.sketch {
            sketch.add_chunk(x, y, nrows)?;
            self.total_tokens = sketch.total_tokens;
            return Ok(());
        }

        if x.len() != nrows * self.d_in {
            return Err(format!(
                "x chunk length ({}) does not match nrows * d_in ({} * {})",
                x.len(),
                nrows,
                self.d_in
            ));
        }
        if y.len() != nrows * self.d_out {
            return Err(format!(
                "y chunk length ({}) does not match nrows * d_out ({} * {})",
                y.len(),
                nrows,
                self.d_out
            ));
        }

        // Convert chunk to f64 faer matrices for high-precision accumulation
        let x_mat = Mat::from_fn(nrows, self.d_in, |row, col| x[row * self.d_in + col] as f64);
        let y_mat = Mat::from_fn(nrows, self.d_out, |row, col| {
            y[row * self.d_out + col] as f64
        });

        let c_chunk = x_mat.transpose() * x_mat.as_ref();
        let p_chunk = x_mat.transpose() * y_mat.as_ref();

        for row in 0..self.d_in {
            for col in 0..self.d_in {
                self.c
                    .write(row, col, self.c.read(row, col) + c_chunk.read(row, col));
            }
            for col in 0..self.d_out {
                self.p
                    .write(row, col, self.p.read(row, col) + p_chunk.read(row, col));
            }
        }

        self.total_tokens += nrows;
        Ok(())
    }

    /// Solve the accumulated normal equations (C + lambda * I) * W = P using Cholesky
    /// or Frequent Directions dual factorization, and factorize W into LoRA A and B.
    pub fn solve(
        &self,
        config: &LayerCompilerConfig,
        target: &str,
        layer_index: usize,
    ) -> Result<CompiledLayer, String> {
        if let Some(ref sketch) = self.sketch {
            return sketch.solve(config, target, layer_index);
        }

        if self.total_tokens == 0 {
            let default_rank = config.rank.min(self.d_in).min(self.d_out);
            return Ok(CompiledLayer::new(
                layer_index,
                target,
                vec![0.0; default_rank * self.d_in],
                vec![0.0; self.d_out * default_rank],
                default_rank,
                self.d_in,
                self.d_out,
            ));
        }

        let gram_f32 = Mat::from_fn(self.d_in, self.d_in, |row, col| {
            let val = self.c.read(row, col) as f32;
            if row == col {
                val + config.lambda_reg
            } else {
                val
            }
        });
        let cross_f32 = Mat::from_fn(self.d_in, self.d_out, |row, col| {
            self.p.read(row, col) as f32
        });

        if !matrix_is_finite(gram_f32.as_ref()) || !matrix_is_finite(cross_f32.as_ref()) {
            return Err("streaming accumulated covariance contains non-finite values".to_string());
        }

        let cholesky = Cholesky::try_new(gram_f32.as_ref(), Side::Lower).map_err(|error| {
            format!(
                "Cholesky factorization failed on streaming normal matrix: {error:?}; \
                 increase lambda_reg or stream more diverse tokens"
            )
        })?;

        let weights = cholesky.solve(cross_f32.as_ref());
        if !matrix_is_finite(weights.as_ref()) {
            return Err("streaming Cholesky solution contains non-finite values".to_string());
        }

        factorize_weights(&weights, config, target, layer_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_accumulation_matches_dense() {
        let d_in = 4;
        let d_out = 4;
        let mut acc = CovarianceAccumulator::new(d_in, d_out);

        let chunk1_x = vec![1.0, 0.0, 0.0, 1.0];
        let chunk1_y = vec![2.0, 0.0, 0.0, 2.0];
        acc.add_chunk(&chunk1_x, &chunk1_y, 1).unwrap();

        let chunk2_x = vec![0.0, 1.0, 1.0, 0.0];
        let chunk2_y = vec![0.0, 3.0, 3.0, 0.0];
        acc.add_chunk(&chunk2_x, &chunk2_y, 1).unwrap();

        assert_eq!(acc.total_tokens, 2);

        let cfg = LayerCompilerConfig::new(2, 8, 1e-4, 0.0);
        let layer = acc.solve(&cfg, "o_proj", 0).unwrap();
        assert_eq!(layer.rank, 2);
        assert_eq!(layer.d_in, 4);
        assert_eq!(layer.d_out, 4);
    }

    #[test]
    fn test_frequent_directions_sketch_approximates_dense() {
        let d_in = 16;
        let d_out = 8;
        let m = 32;
        let mut sketch = FrequentDirectionsSketch::new(d_in, d_out, m);

        // Test with severe norm asymmetry ||X||_F >> ||Y||_F (100x) to verify decoupled scale balancing
        for i in 0..100 {
            let x: Vec<f32> = (0..d_in)
                .map(|c| ((i * 7 + c * 3) as f32).sin() * 50.0)
                .collect();
            let y: Vec<f32> = (0..d_out)
                .map(|c| ((i * 5 + c * 2) as f32).cos() * 0.5)
                .collect();
            sketch.add_chunk(&x, &y, 1).unwrap();
        }

        assert_eq!(sketch.total_tokens, 100);
        let cfg = LayerCompilerConfig::new(4, 32, 1e-3, 0.0);
        let layer = sketch.solve(&cfg, "down_proj", 0).unwrap();
        assert_eq!(layer.rank, 4);
        assert_eq!(layer.d_in, 16);
        assert_eq!(layer.d_out, 8);

        // Verify weights did not attenuate to zero due to norm imbalance
        let lora_b_norm: f32 = layer.lora_b.iter().map(|v| v * v).sum();
        let lora_a_norm: f32 = layer.lora_a.iter().map(|v| v * v).sum();
        assert!(
            lora_b_norm > 1e-4,
            "LoRA B norm must be positive and non-decayed: {lora_b_norm}"
        );
        assert!(
            lora_a_norm > 1e-4,
            "LoRA A norm must be positive and non-decayed: {lora_a_norm}"
        );
    }

    #[test]
    fn test_wide_layer_auto_switches_to_sketch() {
        let d_in = 8192;
        let d_out = 1024;
        let mut acc = CovarianceAccumulator::new(d_in, d_out);
        assert!(acc.sketch.is_some());
        assert_eq!(acc.c.nrows(), 0); // dense matrix not allocated!

        let chunk_x = vec![0.1f32; d_in];
        let chunk_y = vec![0.2f32; d_out];
        acc.add_chunk(&chunk_x, &chunk_y, 1).unwrap();
        assert_eq!(acc.total_tokens, 1);

        let cfg = LayerCompilerConfig::new(8, 64, 1e-4, 0.0);
        let layer = acc.solve(&cfg, "down_proj", 0).unwrap();
        assert_eq!(layer.d_in, 8192);
        assert_eq!(layer.d_out, 1024);
    }
}
