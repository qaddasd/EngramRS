use faer::linalg::solvers::{Cholesky, Qr, SpSolver};
use faer::{Mat, MatRef, Side};

use crate::math::factorize::{factorize_reduced_weights, factorize_weights};
use crate::types::{matrix_is_finite, CompiledLayer, LayerCompilerConfig};

fn f64_matrix_is_finite(matrix: MatRef<'_, f64>) -> bool {
    (0..matrix.nrows()).all(|row| (0..matrix.ncols()).all(|col| matrix.read(row, col).is_finite()))
}

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
    /// Joint X/Y balance factor, computed ONCE at the first compression and
    /// reused for every subsequent compression. Recomputing it per buffer
    /// would rescale previously shrunk rows under a new gamma and break the
    /// FD spectral bound.
    pub gamma: Option<f32>,
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
            gamma: None,
        }
    }

    /// Incrementally add a chunk of activations X_chunk [nrows, d_in] and targets Y_chunk [nrows, d_out].
    pub fn add_chunk(&mut self, x: &[f32], y: &[f32], nrows: usize) -> Result<(), String> {
        if nrows == 0 {
            return Ok(());
        }
        let expected_x_len = nrows.checked_mul(self.d_in).ok_or_else(|| {
            "x chunk dimensions (nrows * d_in) overflow addressable memory".to_string()
        })?;
        if x.len() != expected_x_len {
            return Err(format!(
                "x chunk length ({}) does not match nrows * d_in ({} * {})",
                x.len(),
                nrows,
                self.d_in
            ));
        }
        let expected_y_len = nrows.checked_mul(self.d_out).ok_or_else(|| {
            "y chunk dimensions (nrows * d_out) overflow addressable memory".to_string()
        })?;
        if y.len() != expected_y_len {
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
    ///
    /// \gamma is computed ONCE (at the first compression, from the first buffer
    /// contents) and reused for all subsequent compressions: previously shrunk
    /// rows were compressed under that gamma, so recomputing it per buffer
    /// would break the spectral bound.
    pub fn compress(&mut self) -> Result<(), String> {
        if self.current_rows <= self.m {
            return Ok(());
        }

        let n = self.current_rows;
        let d_total = self.d_in + self.d_out;

        let gamma = match self.gamma {
            Some(fixed) => fixed,
            None => {
                // Compute Frobenius norms of the FIRST compressed buffer to
                // determine the balance multiplier \gamma.
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

                let raw_gamma = if norm_y_sq > 1e-12 {
                    ((norm_x_sq / self.d_in as f64) / (norm_y_sq / self.d_out as f64 + 1e-12))
                        .sqrt() as f32
                } else {
                    1.0f32
                };
                if !raw_gamma.is_finite() {
                    return Err(
                        "Frequent Directions balance factor gamma is non-finite; input buffer contains non-finite values"
                            .to_string(),
                    );
                }
                let fixed = raw_gamma.clamp(1e-6, 1e6);
                self.gamma = Some(fixed);
                fixed
            }
        };

        // Form joint matrix Z = [X | \gamma Y] of shape [n, d_in + d_out]
        let joint_z = Mat::from_fn(n, d_total, |r, c| {
            if c < self.d_in {
                self.buffer_x.read(r, c)
            } else {
                gamma * self.buffer_y.read(r, c - self.d_in)
            }
        });

        // Never silently poison the sketch with NaN/Inf: refuse to compress.
        if !matrix_is_finite(joint_z.as_ref()) {
            return Err(
                "Frequent Directions joint buffer contains non-finite values; refusing to compress"
                    .to_string(),
            );
        }

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
                self.buffer_y
                    .write(i, c, inv_gamma * (shrunk * v.read(self.d_in + c, i)));
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

        let expected_x_len = nrows.checked_mul(self.d_in).ok_or_else(|| {
            "x chunk dimensions (nrows * d_in) overflow addressable memory".to_string()
        })?;
        if x.len() != expected_x_len {
            return Err(format!(
                "x chunk length ({}) does not match nrows * d_in ({} * {})",
                x.len(),
                nrows,
                self.d_in
            ));
        }
        let expected_y_len = nrows.checked_mul(self.d_out).ok_or_else(|| {
            "y chunk dimensions (nrows * d_out) overflow addressable memory".to_string()
        })?;
        if y.len() != expected_y_len {
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

        // Solve entirely in f64: the accumulators are f64, so casting the gram
        // and cross matrices to f32 before Cholesky would throw away the
        // accumulated precision on ill-conditioned data. Only the final weight
        // matrix is cast to f32 for the LoRA factorization.
        let mut gram = self.c.clone();
        for diagonal in 0..self.d_in {
            gram.write(
                diagonal,
                diagonal,
                gram.read(diagonal, diagonal) + f64::from(config.lambda_reg),
            );
        }
        let cross = self.p.clone();

        if !f64_matrix_is_finite(gram.as_ref()) || !f64_matrix_is_finite(cross.as_ref()) {
            return Err("streaming accumulated covariance contains non-finite values".to_string());
        }

        let cholesky = Cholesky::try_new(gram.as_ref(), Side::Lower).map_err(|error| {
            format!(
                "Cholesky factorization failed on streaming normal matrix: {error:?}; \
                 increase lambda_reg or stream more diverse tokens"
            )
        })?;

        let weights_f64 = cholesky.solve(cross.as_ref());
        if !f64_matrix_is_finite(weights_f64.as_ref()) {
            return Err("streaming Cholesky solution contains non-finite values".to_string());
        }
        let weights = Mat::from_fn(self.d_in, self.d_out, |row, col| {
            weights_f64.read(row, col) as f32
        });

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

    fn spectral_norm(matrix: faer::MatRef<'_, f32>) -> f64 {
        let svd = matrix.thin_svd();
        let s = svd.s_diagonal();
        if s.nrows() == 0 {
            0.0
        } else {
            f64::from(s.read(0))
        }
    }

    /// R3(a): 10 chunks of 2m rows (many compressions) must approximate the
    /// dense X^T X and X^T Y within the FD spectral bound ||Z||_F^2 / m.
    #[test]
    fn test_fd_multi_compression_respects_spectral_bound() {
        let d_in = 24;
        let d_out = 8;
        let m = 16;
        let chunk_rows = 2 * m;
        let chunks = 10;
        let n = chunk_rows * chunks;
        let x: Vec<f32> = (0..n * d_in)
            .map(|i| ((i * 7 + 3) as f32).sin() * 2.0)
            .collect();
        let y: Vec<f32> = (0..n * d_out)
            .map(|i| ((i * 5 + 1) as f32).cos() * 0.7)
            .collect();

        let mut sketch = FrequentDirectionsSketch::new(d_in, d_out, m);
        for c in 0..chunks {
            sketch
                .add_chunk(
                    &x[c * chunk_rows * d_in..(c + 1) * chunk_rows * d_in],
                    &y[c * chunk_rows * d_out..(c + 1) * chunk_rows * d_out],
                    chunk_rows,
                )
                .unwrap();
        }
        assert_eq!(sketch.total_tokens, n);
        let gamma = sketch
            .gamma
            .expect("gamma must be fixed after compressions");
        let k = sketch.current_rows;
        assert!(k <= m);

        let bx = sketch.buffer_x.submatrix(0, 0, k, d_in);
        let by = sketch.buffer_y.submatrix(0, 0, k, d_out);
        let sketch_gram = Mat::from_fn(d_in, d_in, |i, j| {
            let mut s = 0.0f64;
            for r in 0..k {
                s += f64::from(bx.read(r, i)) * f64::from(bx.read(r, j));
            }
            s
        });
        let sketch_cross = Mat::from_fn(d_in, d_out, |i, j| {
            let mut s = 0.0f64;
            for r in 0..k {
                s += f64::from(bx.read(r, i)) * f64::from(by.read(r, j));
            }
            s
        });
        let dense_gram = Mat::from_fn(d_in, d_in, |i, j| {
            let mut s = 0.0f64;
            for r in 0..n {
                s += f64::from(x[r * d_in + i]) * f64::from(x[r * d_in + j]);
            }
            s
        });
        let dense_cross = Mat::from_fn(d_in, d_out, |i, j| {
            let mut s = 0.0f64;
            for r in 0..n {
                s += f64::from(x[r * d_in + i]) * f64::from(y[r * d_out + j]);
            }
            s
        });
        let diff_gram = Mat::from_fn(d_in, d_in, |i, j| {
            (dense_gram.read(i, j) - sketch_gram.read(i, j)) as f32
        });
        let diff_cross = Mat::from_fn(d_in, d_out, |i, j| {
            (dense_cross.read(i, j) - sketch_cross.read(i, j)) as f32
        });

        let norm_x_sq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let norm_y_sq: f64 = y.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let gamma_f64 = f64::from(gamma);
        let bound = (norm_x_sq + gamma_f64 * gamma_f64 * norm_y_sq) / m as f64;

        let gram_err = spectral_norm(diff_gram.as_ref());
        // The joint off-diagonal block is gamma * X^T Y, so scale its error.
        let cross_err = gamma_f64 * spectral_norm(diff_cross.as_ref());
        assert!(
            gram_err <= bound * 1.02 + 1e-3,
            "FD gram error {gram_err} exceeds bound {bound}"
        );
        assert!(
            cross_err <= bound * 1.02 + 1e-3,
            "FD cross error {cross_err} exceeds bound {bound}"
        );
    }

    /// R3(b): gamma is computed once and stays constant across compressions,
    /// even when later chunks have very different X/Y norm ratios.
    #[test]
    fn test_fd_gamma_fixed_across_compressions() {
        let d_in = 12;
        let d_out = 4;
        let m = 16;
        let mut sketch = FrequentDirectionsSketch::new(d_in, d_out, m);
        assert!(sketch.gamma.is_none());
        let mut recorded = Vec::new();
        for c in 0..8 {
            let scale = 1.0 + 3.0 * (c % 4) as f32;
            let rows = 2 * m;
            let x: Vec<f32> = (0..rows * d_in)
                .map(|i| ((i * 11 + c * 7 + 1) as f32).sin() * scale)
                .collect();
            let y: Vec<f32> = (0..rows * d_out)
                .map(|i| ((i * 3 + c * 5 + 2) as f32).cos() * 0.1 * scale)
                .collect();
            sketch.add_chunk(&x, &y, rows).unwrap();
            recorded.push(sketch.gamma.unwrap());
        }
        let first = recorded[0];
        assert!(
            recorded.iter().all(|&g| g == first),
            "gamma drifted across compressions: {recorded:?}"
        );
        assert!((1e-6..=1e6).contains(&first));
    }

    /// R3(c): a near-zero Y chunk must select gamma = 1.0 and never produce NaN.
    #[test]
    fn test_fd_near_zero_y_chunk_does_not_produce_nan() {
        let d_in = 8;
        let d_out = 4;
        let m = 16;
        let mut sketch = FrequentDirectionsSketch::new(d_in, d_out, m);
        let rows = 2 * m;
        let x: Vec<f32> = (0..rows * d_in)
            .map(|i| ((i * 7 + 1) as f32).sin())
            .collect();
        let y_zero = vec![0.0f32; rows * d_out];
        sketch.add_chunk(&x, &y_zero, rows).unwrap();
        assert_eq!(sketch.gamma, Some(1.0));

        for c in 0..3 {
            let y: Vec<f32> = (0..rows * d_out)
                .map(|i| ((i * 5 + c * 3 + 2) as f32).cos() * 0.5)
                .collect();
            sketch.add_chunk(&x, &y, rows).unwrap();
        }
        let cfg = LayerCompilerConfig::new(4, 32, 1e-3, 0.0);
        let layer = sketch.solve(&cfg, "down_proj", 0).unwrap();
        assert!(layer
            .lora_a
            .iter()
            .chain(&layer.lora_b)
            .all(|v| v.is_finite()));
    }

    /// R3: non-finite buffer contents must return Err, never poison the sketch.
    #[test]
    fn test_fd_non_finite_buffer_returns_error() {
        let d_in = 8;
        let d_out = 4;
        let m = 16;
        let rows = 2 * m;
        let clean_x: Vec<f32> = (0..rows * d_in)
            .map(|i| ((i * 7 + 1) as f32).sin())
            .collect();
        let y = vec![0.5f32; rows * d_out];

        // NaN on the very first compression (gamma not yet fixed).
        let mut sketch = FrequentDirectionsSketch::new(d_in, d_out, m);
        let mut bad_x = clean_x.clone();
        bad_x[5] = f32::NAN;
        let err = sketch.add_chunk(&bad_x, &y, rows).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");

        // Inf arriving after gamma was fixed: the joint-buffer guard must fire.
        let mut sketch2 = FrequentDirectionsSketch::new(d_in, d_out, m);
        sketch2.add_chunk(&clean_x, &y, rows).unwrap();
        let mut bad_y = y.clone();
        bad_y[3] = f32::INFINITY;
        let err2 = sketch2.add_chunk(&clean_x, &bad_y, rows).unwrap_err();
        assert!(err2.contains("non-finite"), "{err2}");
    }

    /// R3: gamma is clamped to [1e-6, 1e6] for extreme norm asymmetry.
    #[test]
    fn test_fd_gamma_clamped_to_range() {
        let d_in = 8;
        let d_out = 4;
        let m = 16;
        let mut sketch = FrequentDirectionsSketch::new(d_in, d_out, m);
        let rows = 2 * m;
        let x = vec![1e9f32; rows * d_in];
        // ||Y||_F^2 must stay above the 1e-12 near-zero guard while the
        // X/Y norm ratio still forces gamma far above the 1e6 clamp.
        let y = vec![1e-5f32; rows * d_out];
        sketch.add_chunk(&x, &y, rows).unwrap();
        assert_eq!(sketch.gamma, Some(1e6));
    }

    /// R8: size computations must use checked_mul and return Err on overflow.
    #[test]
    fn test_add_chunk_size_overflow_returns_error() {
        let mut dense = CovarianceAccumulator::new(4, 4);
        let err = dense.add_chunk(&[], &[], usize::MAX).unwrap_err();
        assert!(err.contains("overflow"), "{err}");

        let mut sketch = FrequentDirectionsSketch::new(4, 4, 16);
        let err = sketch.add_chunk(&[], &[], usize::MAX).unwrap_err();
        assert!(err.contains("overflow"), "{err}");
    }
}
