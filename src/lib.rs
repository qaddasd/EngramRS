// PyO3 0.22 emits a redundant PyErr conversion under Rust 1.95 clippy.
#![allow(clippy::useless_conversion)]

pub mod deduplication;
pub mod math;
pub mod math_core;
pub mod rope_gqa;
pub mod serializer;
pub mod simd;
pub mod streaming;
pub mod types;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use math_core::{compile_all_targets, LayerCompilerConfig, TargetData};
use numpy::{
    PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serializer::write_adapter;

#[pyfunction]
fn center_and_unpack_bf16_py<'py>(
    py: Python<'py>,
    raw_bf16: PyReadonlyArrayDyn<'py, u16>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = raw_bf16.shape();
    let in_slice = raw_bf16
        .as_slice()
        .map_err(|_| PyValueError::new_err("raw_bf16 must be C-contiguous"))?;

    let out = PyArrayDyn::<f32>::zeros_bound(py, shape, false);
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyValueError::new_err("failed to acquire mutable output slice"))?
    };

    if shape.len() == 2 {
        let nrows = shape[0];
        let d_model = shape[1];
        simd::centering::center_and_unpack_bf16(in_slice, nrows, d_model, out_slice);
    } else if shape.len() == 3 {
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        simd::centering::center_and_unpack_bf16_batched(
            in_slice, batch_size, seq_len, d_model, out_slice,
        );
    } else {
        return Err(PyValueError::new_err(
            "tensor must have 2 or 3 dimensions for sink centering",
        ));
    }
    Ok(out)
}

#[pyfunction]
fn center_f32_py<'py>(
    py: Python<'py>,
    raw_f32: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = raw_f32.shape();
    let in_slice = raw_f32
        .as_slice()
        .map_err(|_| PyValueError::new_err("raw_f32 must be C-contiguous"))?;

    let out = PyArrayDyn::<f32>::zeros_bound(py, shape, false);
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyValueError::new_err("failed to acquire mutable output slice"))?
    };

    if shape.len() == 2 {
        let nrows = shape[0];
        let d_model = shape[1];
        simd::centering::center_f32(in_slice, nrows, d_model, out_slice);
    } else if shape.len() == 3 {
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        simd::centering::center_f32_batched(in_slice, batch_size, seq_len, d_model, out_slice);
    } else {
        return Err(PyValueError::new_err(
            "tensor must have 2 or 3 dimensions for sink centering",
        ));
    }
    Ok(out)
}

#[pyclass(name = "RustStreamingAccumulator")]
pub struct PyStreamingAccumulator {
    inner: streaming::CovarianceAccumulator,
}

type PyLoraPair<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>);

#[pymethods]
impl PyStreamingAccumulator {
    #[new]
    fn new(d_in: usize, d_out: usize) -> Self {
        Self {
            inner: streaming::CovarianceAccumulator::new(d_in, d_out),
        }
    }

    fn add_chunk(
        &mut self,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        if x_shape[0] != y_shape[0] {
            return Err(PyValueError::new_err(format!(
                "token count mismatch: x has {} tokens, y has {} tokens",
                x_shape[0], y_shape[0]
            )));
        }
        let x_slice = x
            .as_slice()
            .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
        let y_slice = y
            .as_slice()
            .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
        self.inner
            .add_chunk(x_slice, y_slice, x_shape[0])
            .map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens
    }

    #[getter]
    fn d_in(&self) -> usize {
        self.inner.d_in
    }

    #[getter]
    fn d_out(&self) -> usize {
        self.inner.d_out
    }

    #[pyo3(signature = (rank, lambda_reg, target=None, layer_index=None))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rank: usize,
        lambda_reg: f32,
        target: Option<&str>,
        layer_index: Option<usize>,
    ) -> PyResult<PyLoraPair<'py>> {
        let config = LayerCompilerConfig::new(rank, 256, lambda_reg, 0.0);
        let compiled = self
            .inner
            .solve(
                &config,
                target.unwrap_or("o_proj"),
                layer_index.unwrap_or(0),
            )
            .map_err(PyRuntimeError::new_err)?;

        let a_arr = PyArray2::<f32>::zeros_bound(py, [compiled.rank, compiled.d_in], false);
        let b_arr = PyArray2::<f32>::zeros_bound(py, [compiled.d_out, compiled.rank], false);

        unsafe {
            a_arr
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&compiled.lora_a);
            b_arr
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&compiled.lora_b);
        }

        Ok((a_arr, b_arr))
    }
}

fn contiguous_slice<'a, 'py>(
    array: &'a PyReadonlyArray2<'py, f32>,
    argument: &str,
    layer: usize,
) -> PyResult<&'a [f32]> {
    array.as_slice().map_err(|_| {
        let shape = array.shape();
        PyValueError::new_err(format!(
            "{argument}[{layer}] with shape [{}, {}] must be C-contiguous; pass a contiguous NumPy array without copying it in the bridge",
            shape[0], shape[1]
        ))
    })
}

/// Compile captured activations and serialize the resulting adapter.
///
/// Supports single-target capture (`hidden_states` + `attention_outputs`) or
/// multi-target capture (`target_inputs` + `target_outputs`).
///
/// The NumPy arrays are borrowed for the duration of compilation. In
/// particular, activation matrices are not copied into Rust-owned vectors.
/// The GIL is intentionally held while the pure Rust compiler runs so all
/// borrowed Python-owned buffers remain soundly alive.
#[pyfunction]
#[pyo3(signature = (
    hidden_states=None,
    attention_outputs=None,
    output_dir=PathBuf::new(),
    null_states=None,
    rank=16,
    max_anchors=256,
    lambda_reg=1e-4,
    lambda_null=1.0,
    base_model_name_or_path="",
    targets=None,
    target_inputs=None,
    target_outputs=None,
    energy_threshold=None,
    min_rank=1,
    max_rank=None,
    depth_scale=None,
    layer_indices=None,
))]
#[allow(clippy::too_many_arguments)]
fn compile_all_layers_to_safetensors(
    hidden_states: Option<Vec<PyReadonlyArray2<'_, f32>>>,
    attention_outputs: Option<Vec<PyReadonlyArray2<'_, f32>>>,
    output_dir: PathBuf,
    null_states: Option<Vec<PyReadonlyArray2<'_, f32>>>,
    rank: usize,
    max_anchors: usize,
    lambda_reg: f32,
    lambda_null: f32,
    base_model_name_or_path: &str,
    targets: Option<Vec<String>>,
    target_inputs: Option<HashMap<String, Vec<PyReadonlyArray2<'_, f32>>>>,
    target_outputs: Option<HashMap<String, Vec<PyReadonlyArray2<'_, f32>>>>,
    energy_threshold: Option<f32>,
    min_rank: usize,
    max_rank: Option<usize>,
    depth_scale: Option<f32>,
    layer_indices: Option<Vec<usize>>,
) -> PyResult<f64> {
    if output_dir.as_os_str().is_empty() {
        return Err(PyValueError::new_err("output_dir must not be empty"));
    }

    let mut all_targets = Vec::new();

    if let (Some(inputs_map), Some(outputs_map)) = (target_inputs.as_ref(), target_outputs.as_ref())
    {
        let target_names: Vec<String> = if let Some(order) = targets {
            order
        } else {
            let mut keys: Vec<String> = inputs_map.keys().cloned().collect();
            keys.sort();
            keys
        };

        if target_names.is_empty() {
            return Err(PyValueError::new_err("targets must not be empty"));
        }

        // Determine layer count from first target
        let first_target = &target_names[0];
        let Some(first_in) = inputs_map.get(first_target) else {
            return Err(PyValueError::new_err(format!(
                "target_inputs is missing entries for target '{first_target}'"
            )));
        };
        let layer_count = first_in.len();
        if layer_count == 0 {
            return Err(PyValueError::new_err(
                "target_inputs must contain at least one layer",
            ));
        }

        if let Some(nulls) = &null_states {
            if nulls.len() != layer_count {
                return Err(PyValueError::new_err(format!(
                    "null_states must contain one matrix per layer ({layer_count}) when provided, got {}",
                    nulls.len()
                )));
            }
        }

        for target in &target_names {
            let Some(in_arrays) = inputs_map.get(target) else {
                return Err(PyValueError::new_err(format!(
                    "target_inputs is missing entries for target '{target}'"
                )));
            };
            let Some(out_arrays) = outputs_map.get(target) else {
                return Err(PyValueError::new_err(format!(
                    "target_outputs is missing entries for target '{target}'"
                )));
            };

            if in_arrays.len() != layer_count {
                return Err(PyValueError::new_err(format!(
                    "target '{target}' inputs has {} layers, expected {layer_count}",
                    in_arrays.len()
                )));
            }
            if out_arrays.len() != layer_count {
                return Err(PyValueError::new_err(format!(
                    "target '{target}' outputs has {} layers, expected {layer_count}",
                    out_arrays.len()
                )));
            }

            for layer in 0..layer_count {
                let in_array = &in_arrays[layer];
                let out_array = &out_arrays[layer];
                let in_shape = in_array.shape();
                let out_shape = out_array.shape();

                if in_shape[0] != out_shape[0] {
                    return Err(PyValueError::new_err(format!(
                        "layer {layer} target '{target}' token count mismatch: input has {} tokens, output has {} tokens",
                        in_shape[0], out_shape[0]
                    )));
                }
                if in_shape[1] == 0 {
                    return Err(PyValueError::new_err(format!(
                        "layer {layer} target '{target}' has d_in=0; input matrix must have non-zero features"
                    )));
                }
                if out_shape[1] == 0 {
                    return Err(PyValueError::new_err(format!(
                        "layer {layer} target '{target}' has d_out=0; output matrix must have non-zero features"
                    )));
                }

                let in_slice = contiguous_slice(in_array, &format!("{target}_inputs"), layer)?;
                let out_slice = contiguous_slice(out_array, &format!("{target}_outputs"), layer)?;
                let null = match &null_states {
                    Some(nulls) => {
                        let null_array = &nulls[layer];
                        let null_shape = null_array.shape();
                        if null_shape[1] == in_shape[1] {
                            Some(contiguous_slice(null_array, "null_states", layer)?)
                        } else {
                            None
                        }
                    }
                    None => None,
                };

                let actual_layer_idx = match &layer_indices {
                    Some(indices) if layer < indices.len() => indices[layer],
                    _ => layer,
                };

                all_targets.push(TargetData {
                    target: target.clone(),
                    layer_index: actual_layer_idx,
                    input: in_slice,
                    output: out_slice,
                    null_states: null,
                    d_in: in_shape[1],
                    d_out: out_shape[1],
                });
            }
        }
    } else if let (Some(hs), Some(ao)) = (hidden_states.as_ref(), attention_outputs.as_ref()) {
        let layer_count = hs.len();
        if layer_count == 0 {
            return Err(PyValueError::new_err(
                "hidden_states must contain at least one layer",
            ));
        }
        if ao.len() != layer_count {
            return Err(PyValueError::new_err(format!(
                "hidden_states and attention_outputs must contain the same number of layers, got {layer_count} and {}",
                ao.len()
            )));
        }
        if let Some(nulls) = &null_states {
            if nulls.len() != layer_count {
                return Err(PyValueError::new_err(format!(
                    "null_states must contain one matrix per layer ({layer_count}) when provided, got {}",
                    nulls.len()
                )));
            }
        }

        for layer in 0..layer_count {
            let hidden_array = &hs[layer];
            let attention_array = &ao[layer];
            let hidden_shape = hidden_array.shape();
            let attention_shape = attention_array.shape();
            if hidden_shape != attention_shape {
                return Err(PyValueError::new_err(format!(
                    "layer {layer} shape mismatch: hidden_states has shape [{}, {}], attention_outputs has shape [{}, {}]; both must be [tokens, d_model]",
                    hidden_shape[0],
                    hidden_shape[1],
                    attention_shape[0],
                    attention_shape[1]
                )));
            }
            if hidden_shape[1] == 0 {
                return Err(PyValueError::new_err(format!(
                    "layer {layer} has d_model=0; activation matrices must have a non-zero second dimension"
                )));
            }

            let hidden = contiguous_slice(hidden_array, "hidden_states", layer)?;
            let attention = contiguous_slice(attention_array, "attention_outputs", layer)?;
            let null = match &null_states {
                Some(nulls) => {
                    let null_array = &nulls[layer];
                    let null_shape = null_array.shape();
                    if null_shape[1] != hidden_shape[1] {
                        return Err(PyValueError::new_err(format!(
                            "layer {layer} shape mismatch: null_states has d_model={}, expected {}; its token count may differ",
                            null_shape[1], hidden_shape[1]
                        )));
                    }
                    Some(contiguous_slice(null_array, "null_states", layer)?)
                }
                None => None,
            };

            let actual_layer_idx = match &layer_indices {
                Some(indices) if layer < indices.len() => indices[layer],
                _ => layer,
            };

            all_targets.push(TargetData {
                target: "o_proj".to_string(),
                layer_index: actual_layer_idx,
                input: hidden,
                output: attention,
                null_states: null,
                d_in: hidden_shape[1],
                d_out: hidden_shape[1],
            });
        }
    } else {
        return Err(PyValueError::new_err(
            "must provide either hidden_states and attention_outputs or target_inputs and target_outputs",
        ));
    }

    let target_layer_count = all_targets
        .iter()
        .map(|t| t.layer_index)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);
    let resolved_depth_scale =
        depth_scale.unwrap_or_else(|| 1.0 / (target_layer_count as f32).max(1.0).sqrt());

    let config = LayerCompilerConfig {
        rank,
        max_anchors,
        lambda_reg,
        lambda_null,
        energy_threshold,
        min_rank,
        max_rank: max_rank.unwrap_or(rank),
        depth_scale: resolved_depth_scale,
    };

    let started = Instant::now();
    let compiled = compile_all_targets(&all_targets, &config).map_err(|error| {
        PyRuntimeError::new_err(format!("Rust layer compilation failed: {error}"))
    })?;
    write_adapter(&output_dir, &compiled, base_model_name_or_path).map_err(|error| {
        PyRuntimeError::new_err(format!(
            "failed to serialize adapter to '{}': {error}",
            output_dir.display()
        ))
    })?;
    Ok(started.elapsed().as_secs_f64() * 1_000.0)
}

#[allow(clippy::too_many_arguments)]
fn rope_py_impl<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    positions: PyReadonlyArray1<'py, i64>,
    base: f64,
    partial_rotary_factor: f64,
    inverse: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = x.shape();
    let (n_tokens, n_heads, head_dim) = (shape[0], shape[1], shape[2]);
    let x_slice = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let pos_slice = positions
        .as_slice()
        .map_err(|_| PyValueError::new_err("positions must be C-contiguous"))?;
    let rotated = if inverse {
        rope_gqa::rope_inverse(
            x_slice,
            pos_slice,
            base,
            partial_rotary_factor,
            n_tokens,
            n_heads,
            head_dim,
        )
    } else {
        rope_gqa::rope_apply(
            x_slice,
            pos_slice,
            base,
            partial_rotary_factor,
            n_tokens,
            n_heads,
            head_dim,
        )
    }
    .map_err(PyValueError::new_err)?;

    let out = PyArray3::<f32>::zeros_bound(py, [n_tokens, n_heads, head_dim], false);
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyValueError::new_err("failed to acquire mutable output slice"))?
    };
    out_slice.copy_from_slice(&rotated);
    Ok(out)
}

/// Apply NeoX half-split RoPE to `x` of shape [n_tokens, n_heads, head_dim]
/// with per-token i64 `positions`, theta `base` and `partial_rotary_factor`
/// in (0, 1].
#[pyfunction]
#[pyo3(signature = (x, positions, base=10000.0, partial_rotary_factor=1.0))]
fn rope_apply_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    positions: PyReadonlyArray1<'py, i64>,
    base: f64,
    partial_rotary_factor: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    rope_py_impl(py, x, positions, base, partial_rotary_factor, false)
}

/// Undo RoPE applied by `rope_apply_py` with the same positions/base/factor.
#[pyfunction]
#[pyo3(signature = (x, positions, base=10000.0, partial_rotary_factor=1.0))]
fn rope_inverse_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    positions: PyReadonlyArray1<'py, i64>,
    base: f64,
    partial_rotary_factor: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    rope_py_impl(py, x, positions, base, partial_rotary_factor, true)
}

/// Expand KV heads for grouped-query attention:
/// [n_tokens, n_kv_heads, head_dim] -> [n_tokens, n_kv_heads * n_rep, head_dim].
/// Query head h maps to KV head h // n_rep (torch-style contiguous groups).
#[pyfunction]
fn repeat_kv_heads_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    n_rep: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if n_rep == 0 {
        return Err(PyValueError::new_err("n_rep must be greater than zero"));
    }
    let shape = x.shape();
    let (n_tokens, n_kv_heads, head_dim) = (shape[0], shape[1], shape[2]);
    let n_query_heads = n_kv_heads
        .checked_mul(n_rep)
        .ok_or_else(|| PyValueError::new_err("n_kv_heads * n_rep overflows addressable memory"))?;
    let kv_slice = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let expanded =
        rope_gqa::repeat_kv_heads(kv_slice, n_tokens, n_query_heads, n_kv_heads, head_dim)
            .map_err(PyValueError::new_err)?;

    let out = PyArray3::<f32>::zeros_bound(py, [n_tokens, n_query_heads, head_dim], false);
    let out_slice = unsafe {
        out.as_slice_mut()
            .map_err(|_| PyValueError::new_err("failed to acquire mutable output slice"))?
    };
    out_slice.copy_from_slice(&expanded);
    Ok(out)
}

#[pymodule]
fn engram_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_all_layers_to_safetensors, module)?)?;
    module.add_function(wrap_pyfunction!(center_and_unpack_bf16_py, module)?)?;
    module.add_function(wrap_pyfunction!(center_f32_py, module)?)?;
    module.add_function(wrap_pyfunction!(rope_apply_py, module)?)?;
    module.add_function(wrap_pyfunction!(rope_inverse_py, module)?)?;
    module.add_function(wrap_pyfunction!(repeat_kv_heads_py, module)?)?;
    module.add_class::<PyStreamingAccumulator>()?;
    Ok(())
}
