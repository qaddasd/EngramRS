//! Legacy facade for math_core providing 100% backward compatibility.
//!
//! Submodules have been decomposed into:
//! - `crate::types` (data structures, config, validation)
//! - `crate::deduplication` (zero-alloc fast hashing & deduplication)
//! - `crate::math::anchors` (SVD & randomized SVD compression)
//! - `crate::math::solvers` (Dual Woodbury & Primal Cholesky)
//! - `crate::math::factorize` (SVD factorization & spectral energy rank allocation)
//! - `crate::math::mlp_solver` (analytical solver for SwiGLU MLP down_proj)

pub use crate::math::{
    compile_all_layers, compile_all_targets, compile_layer_weights, compile_mlp_down_proj,
    compile_target_weights, factorize_dual_via_svd, factorize_reduced_weights, factorize_weights,
    project_layer_null_space, randomized_svd_anchors, randomized_svd_anchors_seeded,
    select_rank_by_energy, solve_dual_and_factorize, solve_primal_and_factorize, svd_anchors,
};
pub use crate::serializer::write_adapter;
pub use crate::types::{
    matrix_is_finite, validate_finite, validate_target, AnchoredMatrices, CompiledLayer,
    LayerCompilerConfig, LayerData, TargetData,
};

#[cfg(test)]
pub mod tests {
    #[allow(unused_imports)]
    pub use crate::math::tests::*;
}
