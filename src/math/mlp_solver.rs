use faer::Mat;

use crate::deduplication::deduplicate_rows;
use crate::math::anchors::svd_anchors;
use crate::math::solvers::{solve_dual_and_factorize, solve_primal_and_factorize};
use crate::types::{validate_target, CompiledLayer, LayerCompilerConfig, TargetData};

/// Closed-form analytical solver for MLP SwiGLU down_proj layers.
///
/// In SwiGLU FFN:
///   H_mlp = silu(W_gate x) * (W_up x)  in R^{N x d_ffn}
///   down_proj output in R^{N x d_model}
///
/// Because d_ffn is typically large (e.g. 11,000 - 14,336 in LLaMA/Qwen),
/// the dual Woodbury formulation:
///   W_delta = H_mlp^T (H_mlp H_mlp^T + lambda I)^-1 Delta_Y
/// computes the exact low-rank correction in R^{N x N} space in milliseconds
/// instead of inverting a massive R^{d_ffn x d_ffn} matrix.
pub fn compile_mlp_down_proj(
    target: &TargetData<'_>,
    config: &LayerCompilerConfig,
) -> Result<CompiledLayer, String> {
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
