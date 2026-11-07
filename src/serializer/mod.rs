pub mod peft_config;
pub mod safetensors;

use std::path::Path;

pub use peft_config::{module_sub_path, target_module_short_name, write_peft_config};
pub use safetensors::{f16_bytes, f32_bytes, write_safetensors, write_safetensors_typed};

use crate::types::CompiledLayer;

/// Write `adapter_model.safetensors` and `adapter_config.json` in PEFT format.
pub fn write_adapter(
    output_dir: &Path,
    layers: &[CompiledLayer],
    base_model_name_or_path: &str,
) -> Result<(), String> {
    write_adapter_typed(output_dir, layers, base_model_name_or_path, false)
}

/// Write adapter with selectable float16 or float32 weight storage.
pub fn write_adapter_typed(
    output_dir: &Path,
    layers: &[CompiledLayer],
    base_model_name_or_path: &str,
    use_f16: bool,
) -> Result<(), String> {
    validate_layers(layers, base_model_name_or_path)?;
    std::fs::create_dir_all(output_dir).map_err(|error| {
        format!(
            "failed to create adapter output directory {}: {error}",
            output_dir.display()
        )
    })?;

    write_safetensors_typed(output_dir, layers, use_f16)?;
    write_peft_config(output_dir, layers, base_model_name_or_path)?;
    Ok(())
}

fn validate_layers(layers: &[CompiledLayer], base_model_name_or_path: &str) -> Result<(), String> {
    let Some(first) = layers.first() else {
        return Err("cannot write an adapter with zero layers".to_string());
    };
    if base_model_name_or_path.trim().is_empty() {
        return Err("base_model_name_or_path must not be empty".to_string());
    }
    for (index, layer) in layers.iter().enumerate() {
        if layer.rank == 0 || layer.d_in == 0 || layer.d_out == 0 {
            return Err("compiled layers must have positive rank, d_in, and d_out".to_string());
        }
        if layer.d_out != first.d_out {
            return Err(format!(
                "layer {index} d_model ({}) differs from layer 0 ({}); a consistent model dimension is required",
                layer.d_out, first.d_out
            ));
        }
        let a_len = layer.rank.checked_mul(layer.d_in).ok_or_else(|| {
            format!("layer {index} lora_A dimensions overflow addressable memory")
        })?;
        let b_len = layer.d_out.checked_mul(layer.rank).ok_or_else(|| {
            format!("layer {index} lora_B dimensions overflow addressable memory")
        })?;
        if layer.lora_a.len() != a_len {
            return Err(format!(
                "layer {index} lora_A length ({}) does not match rank*d_in ({a_len})",
                layer.lora_a.len()
            ));
        }
        if layer.lora_b.len() != b_len {
            return Err(format!(
                "layer {index} lora_B length ({}) does not match d_out*rank ({b_len})",
                layer.lora_b.len()
            ));
        }
        if let Some((flat_index, _)) = layer
            .lora_a
            .iter()
            .chain(&layer.lora_b)
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "layer {index} contains a non-finite LoRA value at combined flat index {flat_index}"
            ));
        }
    }
    Ok(())
}
