use half::{bf16, f16};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::path::Path;

use super::peft_config::module_sub_path;
use crate::types::CompiledLayer;

/// Largest finite magnitude representable in IEEE binary16 (f16).
const F16_MAX: f32 = 65504.0;

/// Weight storage dtype for adapter serialization (internal Rust API).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightDtype {
    F32,
    F16,
    BF16,
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Convert f32 values to little-endian f16 bytes.
///
/// Returns `Err` naming the offending value instead of silently producing
/// infinities when a value is non-finite or its magnitude exceeds the f16
/// maximum of 65504.
pub fn f16_bytes(values: &[f32]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || value.abs() > F16_MAX {
            return Err(format!(
                "value {value} at flat index {index} cannot be stored as f16 \
                 (must be finite with |v| <= {F16_MAX})"
            ));
        }
        let half_val = f16::from_f32(value);
        bytes.extend_from_slice(&half_val.to_le_bytes());
    }
    Ok(bytes)
}

/// Convert f32 values to little-endian bf16 bytes.
///
/// bf16 shares the f32 exponent range, so no overflow check is needed for
/// finite f32 inputs; precision is reduced to 8 significand bits.
pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &value in values {
        bytes.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
    }
    bytes
}

pub fn write_safetensors(output_dir: &Path, layers: &[CompiledLayer]) -> Result<(), String> {
    write_safetensors_typed(output_dir, layers, false)
}

/// Legacy bool-typed writer kept for backward compatibility.
pub fn write_safetensors_typed(
    output_dir: &Path,
    layers: &[CompiledLayer],
    use_f16: bool,
) -> Result<(), String> {
    let dtype = if use_f16 {
        WeightDtype::F16
    } else {
        WeightDtype::F32
    };
    write_safetensors_dtype(output_dir, layers, dtype)
}

/// Write `adapter_model.safetensors` with an explicit storage dtype
/// (F32, F16 or BF16).
pub fn write_safetensors_dtype(
    output_dir: &Path,
    layers: &[CompiledLayer],
    dtype: WeightDtype,
) -> Result<(), String> {
    let mut tensor_entries = Vec::with_capacity(layers.len() * 2);
    for layer in layers {
        let sub_path = module_sub_path(&layer.target);
        let prefix = format!(
            "base_model.model.model.layers.{}.{}.lora_",
            layer.layer_index, sub_path
        );
        let (a_bytes, b_bytes) = match dtype {
            WeightDtype::F32 => (f32_bytes(&layer.lora_a), f32_bytes(&layer.lora_b)),
            WeightDtype::F16 => (f16_bytes(&layer.lora_a)?, f16_bytes(&layer.lora_b)?),
            WeightDtype::BF16 => (bf16_bytes(&layer.lora_a), bf16_bytes(&layer.lora_b)),
        };
        tensor_entries.push((
            format!("{prefix}A.weight"),
            vec![layer.rank, layer.d_in],
            a_bytes,
        ));
        tensor_entries.push((
            format!("{prefix}B.weight"),
            vec![layer.d_out, layer.rank],
            b_bytes,
        ));
    }

    let safetensors_dtype = match dtype {
        WeightDtype::F32 => Dtype::F32,
        WeightDtype::F16 => Dtype::F16,
        WeightDtype::BF16 => Dtype::BF16,
    };
    let mut views = Vec::with_capacity(tensor_entries.len());
    for (name, shape, bytes) in &tensor_entries {
        let view = TensorView::new(safetensors_dtype, shape.clone(), bytes)
            .map_err(|error| format!("failed to construct tensor {name}: {error}"))?;
        views.push((name.clone(), view));
    }

    let adapter_path = output_dir.join("adapter_model.safetensors");
    serialize_to_file(views, &None, &adapter_path).map_err(|error| {
        format!(
            "failed to write adapter weights to {}: {error}",
            adapter_path.display()
        )
    })?;
    Ok(())
}
