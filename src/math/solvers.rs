use faer::linalg::solvers::{Cholesky, Qr, SpSolver};
use faer::{Mat, Side};

use crate::math::factorize::{factorize_reduced_weights, factorize_weights};
use crate::types::{matrix_is_finite, AnchoredMatrices, CompiledLayer, LayerCompilerConfig};

pub fn solve_primal_and_factorize(
    anchored: &AnchoredMatrices,
    null_states: Option<&[f32]>,
    null_rows: usize,
    config: &LayerCompilerConfig,
    target: &str,
    layer_index: usize,
) -> Result<CompiledLayer, String> {
    let d_in = anchored.x.ncols();
    let mut gram = anchored.x.transpose() * anchored.x.as_ref();
    let cross = anchored.x.transpose() * anchored.y.as_ref();

    if null_rows > 0 {
        let null_states = null_states.expect("non-zero null row count requires null states");
        let null = Mat::from_fn(null_rows, d_in, |row, col| null_states[row * d_in + col]);
        let null_gram = null.transpose() * null.as_ref();
        for row in 0..d_in {
            for col in 0..d_in {
                gram.write(
                    row,
                    col,
                    gram.read(row, col) + config.lambda_null * null_gram.read(row, col),
                );
            }
        }
    }
    for diagonal in 0..d_in {
        gram.write(
            diagonal,
            diagonal,
            gram.read(diagonal, diagonal) + config.lambda_reg,
        );
    }
    if !matrix_is_finite(gram.as_ref()) || !matrix_is_finite(cross.as_ref()) {
        return Err("normal equations contain non-finite values; rescale the input".to_string());
    }

    let cholesky = Cholesky::try_new(gram.as_ref(), Side::Lower).map_err(|error| {
        format!(
            "Cholesky factorization failed for the regularized normal matrix: {error:?}; \
             increase lambda_reg or improve input conditioning"
        )
    })?;
    let weights = cholesky.solve(cross.as_ref());
    if !matrix_is_finite(weights.as_ref()) {
        return Err("Cholesky solution contains non-finite values".to_string());
    }
    factorize_weights(&weights, config, target, layer_index)
}

pub fn solve_dual_and_factorize(
    anchored: &AnchoredMatrices,
    null_states: Option<&[f32]>,
    null_rows: usize,
    config: &LayerCompilerConfig,
    target: &str,
    layer_index: usize,
) -> Result<CompiledLayer, String> {
    let x = anchored.x.as_ref();
    let y = anchored.y.as_ref();
    let n = x.nrows();
    let d_in = x.ncols();
    let d_out = y.ncols();
    let p = n + null_rows;
    let scale = config.lambda_null.sqrt();
    let z = Mat::from_fn(p, d_in, |row, col| {
        if row < n {
            x.read(row, col)
        } else {
            let null_states = null_states.expect("non-zero null row count requires null states");
            scale * null_states[(row - n) * d_in + col]
        }
    });
    let rhs = Mat::from_fn(
        p,
        d_out,
        |row, col| {
            if row < n {
                y.read(row, col)
            } else {
                0.0
            }
        },
    );
    let mut gram = z.as_ref() * z.transpose();
    for diagonal in 0..p {
        gram.write(
            diagonal,
            diagonal,
            gram.read(diagonal, diagonal) + config.lambda_reg,
        );
    }
    if !matrix_is_finite(gram.as_ref()) || !matrix_is_finite(rhs.as_ref()) {
        return Err(
            "dual normal equations contain non-finite values; rescale the input".to_string(),
        );
    }
    let cholesky = Cholesky::try_new(gram.as_ref(), Side::Lower).map_err(|error| {
        format!(
            "dual Cholesky factorization failed for the regularized normal matrix: {error:?}; \
             increase lambda_reg or improve input conditioning"
        )
    })?;
    let solved = cholesky.solve(rhs.as_ref());
    if !matrix_is_finite(solved.as_ref()) {
        return Err("dual Cholesky solution contains non-finite values".to_string());
    }

    // Z^T = Q R and W = Z^T (Z Z^T + lambda I)^-1 rhs = Q (R solved).
    let qr = Qr::new(z.transpose());
    let q = qr.compute_thin_q();
    let r = qr.compute_thin_r();
    if matrix_is_finite(q.as_ref()) && matrix_is_finite(r.as_ref()) {
        let reduced = r * solved.as_ref();
        if matrix_is_finite(reduced.as_ref()) {
            return factorize_reduced_weights(&q, &reduced, config, target, layer_index);
        }
    }
    factorize_dual_via_svd(&z, &solved, config, target, layer_index)
}

pub fn factorize_dual_via_svd(
    z: &Mat<f32>,
    solved: &Mat<f32>,
    config: &LayerCompilerConfig,
    target: &str,
    layer_index: usize,
) -> Result<CompiledLayer, String> {
    let p = z.nrows();
    let svd = z.transpose().thin_svd();
    let q = svd.u().to_owned();
    let s = svd.s_diagonal();
    let v = svd.v();
    let r = Mat::from_fn(p, p, |row, col| s.read(row) * v.read(col, row));
    let reduced = r * solved.as_ref();
    if !matrix_is_finite(q.as_ref()) || !matrix_is_finite(reduced.as_ref()) {
        return Err("rank-deficient dual factorization produced non-finite values".to_string());
    }
    factorize_reduced_weights(&q, &reduced, config, target, layer_index)
}
