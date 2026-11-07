use faer::Mat;

use crate::types::{validate_finite, CompiledLayer, LayerCompilerConfig};

/// Select rank dynamically based on singular value energy threshold:
/// \sum_{i=1}^r \sigma_i^2 / \sum \sigma_j^2 >= energy_threshold,
/// bounded by [min_rank, max_rank] and min(d_in, d_out).
pub fn select_rank_by_energy(
    singular_values: &[f32],
    config: &LayerCompilerConfig,
    d_in: usize,
    d_out: usize,
) -> usize {
    let max_possible = singular_values.len().min(d_in).min(d_out);
    if max_possible == 0 {
        return 0;
    }
    match config.energy_threshold {
        Some(threshold) => {
            let total_energy: f64 = singular_values[..max_possible]
                .iter()
                .map(|&s| (s as f64) * (s as f64))
                .sum();
            if total_energy <= 0.0 || !total_energy.is_finite() {
                let min_r = config.min_rank.max(1);
                let max_r = config.max_rank.max(min_r);
                return min_r.clamp(1, max_possible).min(max_r);
            }
            let target_energy = (threshold as f64) * total_energy;
            let mut cumulative = 0.0;
            let mut chosen = max_possible;
            for (idx, &s) in singular_values[..max_possible].iter().enumerate() {
                cumulative += (s as f64) * (s as f64);
                if cumulative >= target_energy {
                    chosen = idx + 1;
                    break;
                }
            }
            let min_r = config.min_rank.max(1);
            let max_r = config.max_rank.max(min_r);
            chosen.clamp(min_r, max_r).min(max_possible)
        }
        None => config.rank.min(max_possible),
    }
}

pub fn factorize_reduced_weights(
    q: &Mat<f32>,
    reduced: &Mat<f32>,
    config: &LayerCompilerConfig,
    target: &str,
    layer_index: usize,
) -> Result<CompiledLayer, String> {
    let d_in = q.nrows();
    let d_out = reduced.ncols();
    let p = reduced.nrows();
    let max_components = p.min(d_in).min(d_out);
    let svd = reduced.as_ref().thin_svd();
    let u = svd.u();
    let s = svd.s_diagonal();
    let v = svd.v();
    let m = s.nrows();

    let mut s_vals = Vec::with_capacity(m);
    for i in 0..m {
        s_vals.push(s.read(i));
    }

    let rank = select_rank_by_energy(&s_vals, config, d_in, d_out);
    if rank == 0 {
        return Err("selected rank must be greater than zero".to_string());
    }
    let components = rank.min(max_components);
    let rotated = q.as_ref() * u.submatrix(0, 0, p, components);
    let mut lora_a = vec![0.0; rank * d_in];
    let mut lora_b = vec![0.0; d_out * rank];
    for component in 0..components {
        let singular_value = s.read(component);
        if !singular_value.is_finite() || singular_value < 0.0 {
            return Err(format!(
                "reduced weight SVD produced invalid singular value {singular_value} at component {component}"
            ));
        }
        let scale = (singular_value * config.depth_scale).sqrt();
        for col in 0..d_in {
            lora_a[component * d_in + col] = scale * rotated.read(col, component);
        }
        for row in 0..d_out {
            lora_b[row * rank + component] = scale * v.read(row, component);
        }
    }
    validate_finite("lora_a", &lora_a)?;
    validate_finite("lora_b", &lora_b)?;
    Ok(CompiledLayer::new(
        layer_index,
        target,
        lora_a,
        lora_b,
        rank,
        d_in,
        d_out,
    ))
}

pub fn factorize_weights(
    weights: &Mat<f32>,
    config: &LayerCompilerConfig,
    target: &str,
    layer_index: usize,
) -> Result<CompiledLayer, String> {
    let d_in = weights.nrows();
    let d_out = weights.ncols();
    if d_in == 0 || d_out == 0 {
        return Err("internal weight matrix has invalid dimensions".to_string());
    }
    let svd = weights.as_ref().thin_svd();
    let u = svd.u();
    let s = svd.s_diagonal();
    let v = svd.v();
    let m = s.nrows();

    let mut s_vals = Vec::with_capacity(m);
    for i in 0..m {
        s_vals.push(s.read(i));
    }

    let rank = select_rank_by_energy(&s_vals, config, d_in, d_out);
    if rank == 0 {
        return Err("selected rank must be greater than zero".to_string());
    }

    let mut scales = Vec::with_capacity(rank);
    for component in 0..rank {
        let singular_value = s.read(component);
        if !singular_value.is_finite() || singular_value < 0.0 {
            return Err(format!(
                "weight SVD produced invalid singular value {singular_value} at component {component}"
            ));
        }
        scales.push((singular_value * config.depth_scale).sqrt());
    }

    let mut lora_a = vec![0.0; rank * d_in];
    let mut lora_b = vec![0.0; d_out * rank];
    for component in 0..rank {
        let scale = scales[component];
        for col in 0..d_in {
            lora_a[component * d_in + col] = scale * u.read(col, component);
        }
        for row in 0..d_out {
            lora_b[row * rank + component] = scale * v.read(row, component);
        }
    }
    validate_finite("lora_a", &lora_a)?;
    validate_finite("lora_b", &lora_b)?;
    Ok(CompiledLayer::new(
        layer_index,
        target,
        lora_a,
        lora_b,
        rank,
        d_in,
        d_out,
    ))
}
