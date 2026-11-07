use faer::{Mat, MatRef};

/// Borrowed, contiguous row-major matrices for one model layer (legacy single-target interface).
#[derive(Clone, Debug)]
pub struct LayerData<'a> {
    pub hidden_states: &'a [f32],
    pub attention_output: &'a [f32],
    pub null_states: Option<&'a [f32]>,
    pub d_model: usize,
}

/// Borrowed, contiguous row-major matrices for a specific target module within a layer.
#[derive(Clone, Debug)]
pub struct TargetData<'a> {
    pub target: String,
    pub layer_index: usize,
    pub input: &'a [f32],
    pub output: &'a [f32],
    pub null_states: Option<&'a [f32]>,
    pub d_in: usize,
    pub d_out: usize,
}

#[derive(Clone, Debug)]
pub struct LayerCompilerConfig {
    pub rank: usize,
    pub max_anchors: usize,
    pub lambda_reg: f32,
    pub lambda_null: f32,
    pub energy_threshold: Option<f32>,
    pub min_rank: usize,
    pub max_rank: usize,
    pub depth_scale: f32,
}

impl LayerCompilerConfig {
    pub fn new(rank: usize, max_anchors: usize, lambda_reg: f32, lambda_null: f32) -> Self {
        Self {
            rank,
            max_anchors,
            lambda_reg,
            lambda_null,
            energy_threshold: None,
            min_rank: 1,
            max_rank: rank,
            depth_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledLayer {
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
    pub rank: usize,
    pub d_model: usize,
    pub target: String,
    pub layer_index: usize,
    pub d_in: usize,
    pub d_out: usize,
}

impl CompiledLayer {
    pub fn new(
        layer_index: usize,
        target: impl Into<String>,
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
        rank: usize,
        d_in: usize,
        d_out: usize,
    ) -> Self {
        Self {
            lora_a,
            lora_b,
            rank,
            d_model: d_out,
            target: target.into(),
            layer_index,
            d_in,
            d_out,
        }
    }
}

impl Default for CompiledLayer {
    fn default() -> Self {
        Self {
            lora_a: Vec::new(),
            lora_b: Vec::new(),
            rank: 0,
            d_model: 0,
            target: "o_proj".to_string(),
            layer_index: 0,
            d_in: 0,
            d_out: 0,
        }
    }
}

pub struct AnchoredMatrices {
    pub x: Mat<f32>,
    pub y: Mat<f32>,
}

pub fn validate_target(
    target: &TargetData<'_>,
    config: &LayerCompilerConfig,
) -> Result<(usize, usize), String> {
    if target.d_in == 0 {
        return Err("d_in must be greater than zero".to_string());
    }
    if target.d_out == 0 {
        return Err("d_out must be greater than zero".to_string());
    }
    if config.rank == 0 {
        return Err("rank must be greater than zero".to_string());
    }
    let max_dim = target.d_in.min(target.d_out);
    if config.energy_threshold.is_none() && config.rank > max_dim {
        return Err(format!(
            "rank ({}) cannot exceed min(d_in, d_out) ({max_dim})",
            config.rank
        ));
    }
    if config.rank > max_dim {
        return Err(format!(
            "rank ({}) cannot exceed min(d_in, d_out) ({max_dim})",
            config.rank
        ));
    }
    if config.max_anchors == 0 {
        return Err("max_anchors must be greater than zero".to_string());
    }
    if !config.lambda_reg.is_finite() || config.lambda_reg <= 0.0 {
        return Err("lambda_reg must be finite and greater than zero".to_string());
    }
    if !config.lambda_null.is_finite() || config.lambda_null < 0.0 {
        return Err("lambda_null must be finite and non-negative".to_string());
    }
    if let Some(threshold) = config.energy_threshold {
        if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
            return Err("energy_threshold must be in (0.0, 1.0]".to_string());
        }
        if config.min_rank == 0 {
            return Err("min_rank must be greater than zero".to_string());
        }
        if config.min_rank > config.max_rank {
            return Err(format!(
                "min_rank ({}) cannot exceed max_rank ({})",
                config.min_rank, config.max_rank
            ));
        }
    }

    let d_in = target.d_in;
    let d_out = target.d_out;
    if target.input.is_empty() {
        return Err("input must contain at least one row".to_string());
    }
    if target.input.len() % d_in != 0 {
        return Err(format!(
            "input length ({}) is not divisible by d_in ({d_in})",
            target.input.len()
        ));
    }
    let n = target.input.len() / d_in;
    let expected_output = n
        .checked_mul(d_out)
        .ok_or_else(|| "output dimensions overflow addressable memory".to_string())?;
    if target.output.len() != expected_output {
        return Err(format!(
            "output length ({}) does not match input row count times d_out ({expected_output})",
            target.output.len()
        ));
    }

    let null_n = match target.null_states {
        Some([]) => {
            return Err("null_states, when provided, must contain at least one row".to_string());
        }
        Some(null) if null.len() % d_in != 0 => {
            return Err(format!(
                "null_states length ({}) is not divisible by d_in ({d_in})",
                null.len()
            ));
        }
        Some(null) => null.len() / d_in,
        None => 0,
    };

    validate_finite("input", target.input)?;
    validate_finite("output", target.output)?;
    if let Some(null) = target.null_states {
        validate_finite("null_states", null)?;
    }
    Ok((n, null_n))
}

pub fn validate_finite(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{name} contains a non-finite value ({value}) at flat index {index}"
        ));
    }
    Ok(())
}

pub fn matrix_is_finite(matrix: MatRef<'_, f32>) -> bool {
    (0..matrix.nrows()).all(|row| (0..matrix.ncols()).all(|col| matrix.read(row, col).is_finite()))
}
