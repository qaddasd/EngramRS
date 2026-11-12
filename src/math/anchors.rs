use faer::linalg::solvers::Qr;
use faer::Mat;

use crate::types::{matrix_is_finite, AnchoredMatrices};

/// Build `k <= max_anchors` right-singular-vector anchors.
pub fn svd_anchors(
    x: &Mat<f32>,
    y: &Mat<f32>,
    max_anchors: usize,
) -> Result<AnchoredMatrices, String> {
    let k = max_anchors.min(x.ncols()).min(x.nrows());
    if k == 0 {
        return Err("anchor compression produced zero anchors".to_string());
    }
    if k < x.nrows().min(x.ncols()) {
        return randomized_svd_anchors(x, y, k);
    }
    let svd = x.as_ref().thin_svd();
    let u = svd.u();
    let s = svd.s_diagonal();
    let v = svd.v();
    if u.nrows() != x.nrows() || u.ncols() < k || v.ncols() < k || s.nrows() < k {
        return Err("faer returned an unexpectedly shaped SVD for anchor compression".to_string());
    }

    let u_k = u.submatrix(0, 0, x.nrows(), k);
    let anchored_y = u_k.transpose() * y.as_ref();
    let anchored_x = Mat::from_fn(k, x.ncols(), |row, col| {
        let singular_value = s.read(row);
        singular_value * v.read(col, row)
    });
    if !matrix_is_finite(anchored_x.as_ref()) || !matrix_is_finite(anchored_y.as_ref()) {
        return Err("SVD anchor compression produced non-finite values".to_string());
    }
    Ok(AnchoredMatrices {
        x: anchored_x,
        y: anchored_y,
    })
}

/// Deterministic xoshiro256** PRNG (Blackman & Vigna), seeded via splitmix64.
///
/// Replaces the previous plain LCG so the Gaussian projection matrix Omega is
/// built from high-quality, isotropic randomness.
struct Xoshiro256 {
    state: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut splitmix = seed;
        let mut state = [0u64; 4];
        for slot in &mut state {
            splitmix = splitmix.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = splitmix;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *slot = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform sample in the open interval (0, 1).
    fn next_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// Standard normal N(0, 1) sample via the Box-Muller transform.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Randomized range-finder SVD anchors with the default deterministic seed.
pub fn randomized_svd_anchors(
    x: &Mat<f32>,
    y: &Mat<f32>,
    k: usize,
) -> Result<AnchoredMatrices, String> {
    randomized_svd_anchors_seeded(x, y, k, 0x9e37_79b9_7f4a_7c15)
}

/// Randomized SVD anchors (Halko, Martinsson & Tropp, 2011) with an explicit
/// deterministic seed.
///
/// Quality improvements over the naive single-pass projection:
/// - Gaussian Omega (Box-Muller on xoshiro256**), not uniform LCG noise.
/// - Oversampling p = 8: `width = min(n, d, k + 8)`, which is never below
///   `k + 1` whenever the matrix dimensions allow it.
/// - `q = 2` power iterations `Y <- A (A^T Y)` with QR re-orthonormalization
///   between steps, which sharply improves accuracy for slowly-decaying
///   spectra.
pub fn randomized_svd_anchors_seeded(
    x: &Mat<f32>,
    y: &Mat<f32>,
    k: usize,
    seed: u64,
) -> Result<AnchoredMatrices, String> {
    let max_width = x.nrows().min(x.ncols());
    if k == 0 || k > max_width {
        return Err(format!(
            "randomized anchor rank k={k} must be in 1..={max_width} for a {}x{} matrix",
            x.nrows(),
            x.ncols()
        ));
    }
    // k <= max_width guarantees width >= k, and width >= k + 1 whenever
    // max_width >= k + 1 (oversampling never collapses to zero).
    let width = (k + 8).min(max_width);
    let mut rng = Xoshiro256::new(seed);
    let omega = Mat::from_fn(x.ncols(), width, |_, _| rng.next_gaussian());
    let mut sampled = x.as_ref() * omega.as_ref();
    if !matrix_is_finite(sampled.as_ref()) {
        return Err("randomized anchor projection produced non-finite values".to_string());
    }

    // q = 2 power iterations with QR re-orthonormalization between steps.
    const POWER_ITERATIONS: usize = 2;
    for _ in 0..POWER_ITERATIONS {
        let q_y = Qr::new(sampled.as_ref()).compute_thin_q();
        let projected = x.transpose() * q_y.as_ref();
        if !matrix_is_finite(projected.as_ref()) {
            return Err("randomized power iteration produced non-finite values".to_string());
        }
        let q_z = Qr::new(projected.as_ref()).compute_thin_q();
        sampled = x.as_ref() * q_z.as_ref();
        if !matrix_is_finite(sampled.as_ref()) {
            return Err("randomized power iteration produced non-finite values".to_string());
        }
    }

    let q = Qr::new(sampled.as_ref()).compute_thin_q();
    let reduced = q.transpose() * x.as_ref();
    if !matrix_is_finite(reduced.as_ref()) {
        return Err("randomized anchor reduction produced non-finite values".to_string());
    }
    let svd = reduced.as_ref().thin_svd();
    let u_reduced = svd.u();
    let s = svd.s_diagonal();
    let v = svd.v();
    if u_reduced.ncols() < k || v.ncols() < k || s.nrows() < k {
        return Err("faer returned an unexpectedly shaped randomized SVD".to_string());
    }
    let u = q.as_ref() * u_reduced.submatrix(0, 0, width, k);
    let anchored_y = u.transpose() * y.as_ref();
    let anchored_x = Mat::from_fn(k, x.ncols(), |row, col| s.read(row) * v.read(col, row));
    if !matrix_is_finite(anchored_x.as_ref()) || !matrix_is_finite(anchored_y.as_ref()) {
        return Err("randomized SVD anchors contain non-finite values".to_string());
    }
    Ok(AnchoredMatrices {
        x: anchored_x,
        y: anchored_y,
    })
}
