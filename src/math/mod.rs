pub mod anchors;
pub mod factorize;
pub mod mlp_solver;
pub mod solvers;

use faer::Mat;
use rayon::prelude::*;

use crate::deduplication::deduplicate_rows;
pub use anchors::{randomized_svd_anchors, randomized_svd_anchors_seeded, svd_anchors};
pub use factorize::{factorize_reduced_weights, factorize_weights, select_rank_by_energy};
pub use mlp_solver::compile_mlp_down_proj;
pub use solvers::{factorize_dual_via_svd, solve_dual_and_factorize, solve_primal_and_factorize};

use crate::types::{validate_target, CompiledLayer, LayerCompilerConfig, LayerData, TargetData};

/// Compile a single target module (e.g. `o_proj` or `down_proj`) into an analytical LoRA adapter.
pub fn compile_target_weights(
    target: &TargetData<'_>,
    config: &LayerCompilerConfig,
) -> Result<CompiledLayer, String> {
    if target.target == "down_proj" || target.target == "mlp.down_proj" {
        return compile_mlp_down_proj(target, config);
    }

    let (n, null_n) = validate_target(target, config)?;
    let d_in = target.d_in;
    let d_out = target.d_out;

    let x = Mat::from_fn(n, d_in, |row, col| target.input[row * d_in + col]);
    let y = Mat::from_fn(n, d_out, |row, col| target.output[row * d_out + col]);
    let deduplicated = deduplicate_rows(&x, &y);
    if deduplicated.x.nrows() == 0 {
        let default_rank = config.rank.min(d_in).min(d_out);
        return Ok(CompiledLayer::new(
            target.layer_index,
            &target.target,
            vec![0.0; default_rank * d_in],
            vec![0.0; d_out * default_rank],
            default_rank,
            d_in,
            d_out,
        ));
    }
    let anchored = if deduplicated.x.nrows() > config.max_anchors {
        svd_anchors(&deduplicated.x, &deduplicated.y, config.max_anchors)?
    } else {
        deduplicated
    };

    let null_rows = if config.lambda_null > 0.0 { null_n } else { 0 };
    if anchored.x.nrows() + null_rows < d_in {
        solve_dual_and_factorize(
            &anchored,
            target.null_states,
            null_rows,
            config,
            &target.target,
            target.layer_index,
        )
    } else {
        solve_primal_and_factorize(
            &anchored,
            target.null_states,
            null_rows,
            config,
            &target.target,
            target.layer_index,
        )
    }
}

/// Compile one layer into an analytical LoRA approximation of its state map (legacy single-target interface).
pub fn compile_layer_weights(
    layer: &LayerData<'_>,
    config: &LayerCompilerConfig,
) -> Result<CompiledLayer, String> {
    let target = TargetData {
        target: "o_proj".to_string(),
        layer_index: 0,
        input: layer.hidden_states,
        output: layer.attention_output,
        null_states: layer.null_states,
        d_in: layer.d_model,
        d_out: layer.d_model,
    };
    compile_target_weights(&target, config)
}

/// Compile a list of target modules across layers in parallel.
pub fn compile_all_targets(
    targets: &[TargetData<'_>],
    config: &LayerCompilerConfig,
) -> Result<Vec<CompiledLayer>, String> {
    targets
        .par_iter()
        .map(|target| {
            compile_target_weights(target, config).map_err(|error| {
                format!(
                    "layer {} target {}: {error}",
                    target.layer_index, target.target
                )
            })
        })
        .collect()
}

/// Compile layers in parallel (legacy single-target interface).
pub fn compile_all_layers(
    layers: &[LayerData<'_>],
    config: &LayerCompilerConfig,
) -> Result<Vec<CompiledLayer>, String> {
    let targets: Vec<TargetData<'_>> = layers
        .iter()
        .enumerate()
        .map(|(index, layer)| TargetData {
            target: "o_proj".to_string(),
            layer_index: index,
            input: layer.hidden_states,
            output: layer.attention_output,
            null_states: layer.null_states,
            d_in: layer.d_model,
            d_out: layer.d_model,
        })
        .collect();
    compile_all_targets(&targets, config)
}

/// Project a compiled layer's lora_a into the orthogonal null-space of null_states.
///
/// For any vector x in the row-space of null_states, x * lora_a^T = 0 to machine precision (< 1e-6).
#[allow(clippy::needless_range_loop)]
pub fn project_layer_null_space(
    layer: &mut CompiledLayer,
    null_states: &[f32],
) -> Result<(), String> {
    let d = layer.d_in;
    if null_states.is_empty() || null_states.len() % d != 0 {
        return Err("null_states length must be a non-zero multiple of d_in".to_string());
    }
    let null_rows = null_states.len() / d;
    let null = Mat::from_fn(null_rows, d, |row, col| null_states[row * d + col]);
    let svd = null.as_ref().thin_svd();
    let s = svd.s_diagonal();
    let v = svd.v();
    let s_max = if s.nrows() > 0 { s.read(0) } else { 0.0 };
    let threshold = s_max * 1.0e-5;

    let mut active_cols = Vec::new();
    for col in 0..s.nrows() {
        if s.read(col) > threshold {
            active_cols.push(col);
        }
    }
    if active_cols.is_empty() {
        return Ok(());
    }

    // For each row in lora_a, project out components in range(V_r) using f64 precision
    // and two-pass projection to eliminate single-precision cancellation error.
    for rank_idx in 0..layer.rank {
        let row_start = rank_idx * d;
        let mut row_vec: Vec<f64> = (0..d)
            .map(|col| layer.lora_a[row_start + col] as f64)
            .collect();
        for _ in 0..2 {
            for &v_col in &active_cols {
                let mut dot = 0.0f64;
                for j in 0..d {
                    dot += row_vec[j] * (v.read(j, v_col) as f64);
                }
                for j in 0..d {
                    row_vec[j] -= dot * (v.read(j, v_col) as f64);
                }
            }
        }
        for col in 0..d {
            layer.lora_a[row_start + col] = row_vec[col] as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::types::AnchoredMatrices;
    use faer::linalg::solvers::{Cholesky, SpSolver};
    use faer::{MatRef, Side};

    fn max_abs_difference(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> f32 {
        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
        (0..a.nrows())
            .flat_map(|row| (0..a.ncols()).map(move |col| (row, col)))
            .map(|(row, col)| (a.read(row, col) - b.read(row, col)).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn full_rank_svd_anchors_preserve_normal_equations() {
        let n = 24;
        let d = 6;
        let x = Mat::from_fn(n, d, |row, col| {
            (((row * 17 + col * 31 + 3) % 29) as f32 - 14.0) / 11.0
        });
        let y = Mat::from_fn(n, d, |row, col| {
            (((row * 7 + col * 13 + 5) % 23) as f32 - 11.0) / 9.0
        });
        let anchored = svd_anchors(&x, &y, d).unwrap();
        assert_eq!(anchored.x.nrows(), d);

        let expected_gram = x.transpose() * x.as_ref();
        let actual_gram = anchored.x.transpose() * anchored.x.as_ref();
        let expected_cross = x.transpose() * y.as_ref();
        let actual_cross = anchored.x.transpose() * anchored.y.as_ref();
        assert!(max_abs_difference(expected_gram.as_ref(), actual_gram.as_ref()) < 2.0e-4);
        assert!(max_abs_difference(expected_cross.as_ref(), actual_cross.as_ref()) < 2.0e-4);
    }

    #[test]
    fn truncated_svd_anchors_respect_limit() {
        let x = Mat::from_fn(20, 7, |row, col| ((row + 2 * col) % 11) as f32 / 10.0);
        let y = Mat::from_fn(20, 7, |row, col| ((2 * row + col) % 13) as f32 / 12.0);
        let anchored = svd_anchors(&x, &y, 3).unwrap();
        assert_eq!(anchored.x.nrows(), 3);
        assert_eq!(anchored.y.nrows(), 3);
        assert_eq!(anchored.x.ncols(), 7);
        assert_eq!(anchored.y.ncols(), 7);
    }

    #[test]
    fn cholesky_solves_regularized_system() {
        let d = 4;
        let x = Mat::from_fn(12, d, |row, col| {
            let basis = if row % d == col { 1.0 } else { 0.0 };
            basis + 0.01 * ((row * 7 + col * 3 + 1) as f32).sin()
        });
        let expected = Mat::from_fn(d, d, |row, col| (row + 1) as f32 * 0.2 - col as f32 * 0.07);
        let y = x.as_ref() * expected.as_ref();
        let anchored = AnchoredMatrices { x, y };
        let mut gram = anchored.x.transpose() * anchored.x.as_ref();
        for i in 0..d {
            gram.write(i, i, gram.read(i, i) + 1.0e-6);
        }
        let cross = anchored.x.transpose() * anchored.y.as_ref();
        let solved = Cholesky::try_new(gram.as_ref(), Side::Lower)
            .unwrap()
            .solve(cross.as_ref());
        let difference = max_abs_difference(solved.as_ref(), expected.as_ref());
        assert!(
            difference < 1.0e-4,
            "Cholesky solution differs from expected weights by {difference}"
        );
    }

    #[test]
    fn spectral_energy_allocation_selects_expected_rank() {
        let s = [10.0, 5.0, 1.0, 0.5, 0.1];
        let mut cfg = LayerCompilerConfig::new(5, 10, 1e-4, 0.0);
        cfg.energy_threshold = Some(0.95);
        cfg.min_rank = 1;
        cfg.max_rank = 4;
        let rank = select_rank_by_energy(&s, &cfg, 16, 16);
        assert_eq!(rank, 2);

        cfg.min_rank = 3;
        let rank_clamped = select_rank_by_energy(&s, &cfg, 16, 16);
        assert_eq!(rank_clamped, 3);
    }
}
