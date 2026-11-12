use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

use crate::types::CompiledLayer;

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub struct AdapterConfig<'a> {
    pub base_model_name_or_path: &'a str,
    pub bias: &'static str,
    pub fan_in_fan_out: bool,
    pub inference_mode: bool,
    pub init_lora_weights: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers_to_transform: Option<Vec<usize>>,
    pub lora_alpha: usize,
    pub lora_dropout: f32,
    pub modules_to_save: Option<Vec<String>>,
    pub peft_type: &'static str,
    pub r: usize,
    pub rank_pattern: HashMap<String, usize>,
    pub alpha_pattern: HashMap<String, usize>,
    pub target_modules: Vec<String>,
    pub task_type: Option<&'static str>,
}

pub fn module_sub_path(target: &str) -> String {
    match target {
        "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("self_attn.{target}"),
        "gate_proj" | "up_proj" | "down_proj" => format!("mlp.{target}"),
        other => other.to_string(),
    }
}

pub fn target_module_short_name(target: &str) -> String {
    match target {
        "self_attn.o_proj" | "o_proj" => "o_proj".to_string(),
        "mlp.down_proj" | "down_proj" => "down_proj".to_string(),
        other => {
            if let Some((_, last)) = other.rsplit_once('.') {
                last.to_string()
            } else {
                other.to_string()
            }
        }
    }
}

pub fn write_peft_config(
    output_dir: &Path,
    layers: &[CompiledLayer],
    base_model_name_or_path: &str,
) -> Result<(), String> {
    let mut target_modules: Vec<String> = Vec::new();
    for layer in layers {
        let short = target_module_short_name(&layer.target);
        if !target_modules.contains(&short) {
            target_modules.push(short);
        }
    }

    let max_rank = layers.iter().map(|l| l.rank).max().unwrap_or(1);
    let ranks_differ = layers.iter().any(|l| l.rank != max_rank);
    let mut rank_pattern = HashMap::new();
    let mut alpha_pattern = HashMap::new();
    if ranks_differ {
        for layer in layers {
            let sub_path = module_sub_path(&layer.target);
            let pattern_key = format!("layers.{}.{}", layer.layer_index, sub_path);
            rank_pattern.insert(pattern_key.clone(), layer.rank);
            alpha_pattern.insert(pattern_key, layer.rank);
        }
    }

    let mut unique_layers: Vec<usize> = layers.iter().map(|l| l.layer_index).collect();
    unique_layers.sort_unstable();
    unique_layers.dedup();
    let layers_to_transform = if !unique_layers.is_empty() {
        Some(unique_layers)
    } else {
        None
    };

    let config = AdapterConfig {
        base_model_name_or_path,
        bias: "none",
        fan_in_fan_out: false,
        inference_mode: true,
        init_lora_weights: true,
        layers_to_transform,
        lora_alpha: max_rank,
        lora_dropout: 0.0,
        modules_to_save: None,
        peft_type: "LORA",
        r: max_rank,
        rank_pattern,
        alpha_pattern,
        target_modules,
        task_type: Some("CAUSAL_LM"),
    };

    let config_path = output_dir.join("adapter_config.json");
    let config_file = std::fs::File::create(&config_path).map_err(|error| {
        format!(
            "failed to create adapter configuration {}: {error}",
            config_path.display()
        )
    })?;
    serde_json::to_writer_pretty(config_file, &config).map_err(|error| {
        format!(
            "failed to write adapter configuration {}: {error}",
            config_path.display()
        )
    })?;
    Ok(())
}
