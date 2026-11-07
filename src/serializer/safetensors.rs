use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::path::Path;

use super::peft_config::module_sub_path;
use crate::types::CompiledLayer;

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn write_safetensors(output_dir: &Path, layers: &[CompiledLayer]) -> Result<(), String> {
    let mut tensor_entries = Vec::with_capacity(layers.len() * 2);
    for layer in layers {
        let sub_path = module_sub_path(&layer.target);
        let prefix = format!(
            "base_model.model.model.layers.{}.{}.lora_",
            layer.layer_index, sub_path
        );
        tensor_entries.push((
            format!("{prefix}A.weight"),
            vec![layer.rank, layer.d_in],
            f32_bytes(&layer.lora_a),
        ));
        tensor_entries.push((
            format!("{prefix}B.weight"),
            vec![layer.d_out, layer.rank],
            f32_bytes(&layer.lora_b),
        ));
    }

    let mut views = Vec::with_capacity(tensor_entries.len());
    for (name, shape, bytes) in &tensor_entries {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
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
