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

pub fn randomized_svd_anchors(
    x: &Mat<f32>,
    y: &Mat<f32>,
    k: usize,
) -> Result<AnchoredMatrices, String> {
    let width = (k + 8).min(x.nrows()).min(x.ncols());
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let omega = Mat::from_fn(x.ncols(), width, |_, _| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 40) as f32 / ((1_u64 << 24) - 1) as f32;
        2.0 * unit - 1.0
    });
    let sampled = x.as_ref() * omega.as_ref();
    if !matrix_is_finite(sampled.as_ref()) {
        return Err("randomized anchor projection produced non-finite values".to_string());
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
