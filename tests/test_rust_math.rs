use engram_rs::math_core::{
    compile_all_layers, compile_all_targets, compile_layer_weights, compile_target_weights,
    project_layer_null_space, CompiledLayer, LayerCompilerConfig, LayerData, TargetData,
};
use engram_rs::serializer::write_adapter;
use safetensors::tensor::{Dtype, SafeTensors};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / ((1_u64 << 24) - 1) as f32;
        2.0 * unit - 1.0
    }
}

fn config(rank: usize, max_anchors: usize) -> LayerCompilerConfig {
    LayerCompilerConfig {
        rank,
        max_anchors,
        lambda_reg: 1.0e-5,
        lambda_null: 0.0,
        energy_threshold: None,
        min_rank: 1,
        max_rank: rank,
        depth_scale: 1.0,
    }
}

fn low_rank_data(n: usize, d: usize, rank: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();
    let a: Vec<f32> = (0..d * rank).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..d * rank).map(|_| rng.next_f32()).collect();
    let mut y = vec![0.0; n * d];
    for row in 0..n {
        for component in 0..rank {
            let mut projected = 0.0;
            for inner in 0..d {
                projected += x[row * d + inner] * a[inner * rank + component];
            }
            for col in 0..d {
                y[row * d + col] += projected * b[col * rank + component];
            }
        }
    }
    (x, y)
}

fn apply_lora(x: &[f32], n: usize, layer: &CompiledLayer) -> Vec<f32> {
    let d = layer.d_model;
    let rank = layer.rank;
    assert_eq!(x.len(), n * d);
    let mut result = vec![0.0; n * d];
    let mut projected = vec![0.0; rank];
    for row in 0..n {
        projected.fill(0.0);
        for (component, projection) in projected.iter_mut().enumerate() {
            for inner in 0..d {
                *projection += x[row * d + inner] * layer.lora_a[component * d + inner];
            }
        }
        for col in 0..d {
            for (component, projection) in projected.iter().enumerate() {
                result[row * d + col] += projection * layer.lora_b[col * rank + component];
            }
        }
    }
    result
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn frobenius_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn svd_anchor_compression_reconstructs_low_rank_target() {
    let n = 96;
    let d = 8;
    let rank = 3;
    let (x, y) = low_rank_data(n, d, rank, 42);
    let layer = LayerData {
        hidden_states: &x,
        attention_output: &y,
        null_states: None,
        d_model: d,
    };

    // N > max_anchors, while retaining all d right singular components makes
    // the compressed normal equations equivalent to the original system.
    let compressed = compile_layer_weights(&layer, &config(rank, d)).unwrap();
    let uncompressed = compile_layer_weights(&layer, &config(rank, n)).unwrap();
    let predicted = apply_lora(&x, n, &compressed);
    let baseline = apply_lora(&x, n, &uncompressed);
    assert_eq!(compressed.lora_a.len(), rank * d);
    assert_eq!(compressed.lora_b.len(), d * rank);
    assert!(
        cosine(&predicted, &y) >= 0.999,
        "cosine={}",
        cosine(&predicted, &y)
    );
    assert!(cosine(&predicted, &baseline) >= 0.999);
}

#[test]
fn cholesky_solution_recovers_full_rank_linear_map() {
    let n = 80;
    let d = 7;
    let (x, y) = low_rank_data(n, d, d, 7);
    let compiled = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: None,
            d_model: d,
        },
        &config(d, n),
    )
    .unwrap();
    let predicted = apply_lora(&x, n, &compiled);
    assert!(cosine(&predicted, &y) >= 0.999);
}

#[test]
fn dual_solution_reconstructs_low_rank_map() {
    let n = 5;
    let d = 8;
    let rank = 3;
    let (x, y) = low_rank_data(n, d, rank, 73);
    let compiled = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: None,
            d_model: d,
        },
        &config(rank, n),
    )
    .unwrap();
    let predicted = apply_lora(&x, n, &compiled);
    assert!(
        cosine(&predicted, &y) >= 0.98,
        "cosine={}",
        cosine(&predicted, &y)
    );
}

#[test]
fn dual_solver_handles_repeated_rank_deficient_rows() {
    let n = 6;
    let d = 8;
    let rank = 2;
    let (base_x, base_y) = low_rank_data(3, d, rank, 79);
    let mut x = vec![0.0; n * d];
    let mut y = vec![0.0; n * d];
    for row in 0..n {
        let source = row % 3;
        x[row * d..(row + 1) * d].copy_from_slice(&base_x[source * d..(source + 1) * d]);
        y[row * d..(row + 1) * d].copy_from_slice(&base_y[source * d..(source + 1) * d]);
    }
    let compiled = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: None,
            d_model: d,
        },
        &config(rank, n),
    )
    .unwrap();
    let predicted = apply_lora(&x, n, &compiled);
    assert!(
        cosine(&predicted, &y) >= 0.98,
        "cosine={}",
        cosine(&predicted, &y)
    );
}

#[test]
fn compatible_null_states_have_low_leakage() {
    let n = 48;
    let null_n = 32;
    let d = 8;
    let active = 4;
    let rank = 2;
    let mut rng = Lcg::new(91);
    let mut x = vec![0.0; n * d];
    let mut null = vec![0.0; null_n * d];
    for row in 0..n {
        for col in 0..active {
            x[row * d + col] = rng.next_f32();
        }
    }
    for row in 0..null_n {
        for col in active..d {
            null[row * d + col] = rng.next_f32();
        }
    }
    let output_basis: Vec<f32> = (0..d * rank).map(|_| rng.next_f32()).collect();
    let mut y = vec![0.0; n * d];
    for row in 0..n {
        for component in 0..rank {
            let projected = x[row * d + component];
            for col in 0..d {
                y[row * d + col] += projected * output_basis[col * rank + component];
            }
        }
    }

    let mut compiler_config = config(rank, d);
    compiler_config.lambda_null = 25.0;
    let compiled = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: Some(&null),
            d_model: d,
        },
        &compiler_config,
    )
    .unwrap();
    let predicted = apply_lora(&x, n, &compiled);
    let null_prediction = apply_lora(&null, null_n, &compiled);
    let leakage = frobenius_norm(&null_prediction) / frobenius_norm(&y);
    assert!(
        cosine(&predicted, &y) >= 0.98,
        "cosine={}",
        cosine(&predicted, &y)
    );
    assert!(leakage <= 0.02, "null leakage={leakage}");
}

#[test]
fn dual_null_space_penalty_has_low_leakage() {
    let n = 4;
    let null_n = 3;
    let d = 12;
    let active = 2;
    let rank = 2;
    let mut rng = Lcg::new(117);
    let mut x = vec![0.0; n * d];
    let mut null = vec![0.0; null_n * d];
    for row in 0..n {
        for col in 0..active {
            x[row * d + col] = rng.next_f32();
        }
    }
    for row in 0..null_n {
        for col in active..d {
            null[row * d + col] = rng.next_f32();
        }
    }
    let mut y = vec![0.0; n * d];
    for row in 0..n {
        for col in 0..d {
            y[row * d + col] =
                x[row * d] * (col + 1) as f32 * 0.1 + x[row * d + 1] * (col + 2) as f32 * -0.05;
        }
    }
    let mut compiler_config = config(rank, n);
    compiler_config.lambda_null = 25.0;
    let compiled = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: Some(&null),
            d_model: d,
        },
        &compiler_config,
    )
    .unwrap();
    let predicted = apply_lora(&x, n, &compiled);
    let null_prediction = apply_lora(&null, null_n, &compiled);
    let leakage = frobenius_norm(&null_prediction) / frobenius_norm(&y);
    assert!(
        cosine(&predicted, &y) >= 0.98,
        "cosine={}",
        cosine(&predicted, &y)
    );
    assert!(leakage <= 0.02, "null leakage={leakage}");
}

#[test]
fn zero_anchor_rows_compile_to_zero_adapter() {
    let x = vec![0.0; 4 * 8];
    let y = vec![1.0; 4 * 8];
    let compiled = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: None,
            d_model: 8,
        },
        &config(2, 4),
    )
    .unwrap();
    assert!(compiled.lora_a.iter().all(|value| *value == 0.0));
    assert!(compiled.lora_b.iter().all(|value| *value == 0.0));
}

#[test]
fn invalid_dimensions_and_parameters_return_errors() {
    let d = 4;
    let x = vec![0.25; 6 * d];
    let y = vec![0.5; 6 * d];
    let valid = LayerData {
        hidden_states: &x,
        attention_output: &y,
        null_states: None,
        d_model: d,
    };
    assert!(compile_layer_weights(&valid, &config(0, 8)).is_err());
    assert!(compile_layer_weights(&valid, &config(d + 1, 8)).is_err());
    assert!(compile_layer_weights(&valid, &config(2, 0)).is_err());

    for bad_lambda in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut bad = config(2, 8);
        bad.lambda_reg = bad_lambda;
        assert!(compile_layer_weights(&valid, &bad).is_err());
    }
    for bad_lambda in [-1.0, f32::NAN, f32::NEG_INFINITY] {
        let mut bad = config(2, 8);
        bad.lambda_null = bad_lambda;
        assert!(compile_layer_weights(&valid, &bad).is_err());
    }

    let short_x = &x[..x.len() - 1];
    assert!(compile_layer_weights(
        &LayerData {
            hidden_states: short_x,
            ..valid
        },
        &config(2, 8)
    )
    .is_err());
    let short_y = &y[..y.len() - 1];
    assert!(compile_layer_weights(
        &LayerData {
            attention_output: short_y,
            ..valid
        },
        &config(2, 8)
    )
    .is_err());
    assert!(compile_layer_weights(
        &LayerData {
            hidden_states: &[],
            attention_output: &[],
            null_states: None,
            d_model: d,
        },
        &config(2, 8)
    )
    .is_err());
    assert!(compile_layer_weights(
        &LayerData {
            d_model: 0,
            ..valid
        },
        &config(2, 8)
    )
    .is_err());
    assert!(compile_layer_weights(
        &LayerData {
            null_states: Some(&[]),
            ..valid
        },
        &config(2, 8)
    )
    .is_err());
    assert!(compile_layer_weights(
        &LayerData {
            null_states: Some(&x[..d - 1]),
            ..valid
        },
        &config(2, 8)
    )
    .is_err());

    let mut non_finite = x.clone();
    non_finite[3] = f32::NAN;
    assert!(compile_layer_weights(
        &LayerData {
            hidden_states: &non_finite,
            ..valid
        },
        &config(2, 8)
    )
    .is_err());
}

#[test]
fn parallel_compilation_preserves_layer_order_and_values() {
    let d = 5;
    let rank = 2;
    let owned: Vec<_> = (0..12)
        .map(|index| low_rank_data(30, d, rank, 100 + index))
        .collect();
    let layers: Vec<_> = owned
        .iter()
        .map(|(x, y)| LayerData {
            hidden_states: x,
            attention_output: y,
            null_states: None,
            d_model: d,
        })
        .collect();
    let compiler_config = config(rank, d);
    let parallel = compile_all_layers(&layers, &compiler_config).unwrap();
    let sequential: Vec<_> = layers
        .iter()
        .map(|layer| compile_layer_weights(layer, &compiler_config).unwrap())
        .collect();
    assert_eq!(parallel.len(), layers.len());
    for (index, (parallel, sequential)) in parallel.iter().zip(&sequential).enumerate() {
        assert_eq!(parallel.rank, rank, "wrong layer at index {index}");
        assert_eq!(parallel.d_model, d);
        for (left, right) in parallel
            .lora_a
            .iter()
            .chain(&parallel.lora_b)
            .zip(sequential.lora_a.iter().chain(&sequential.lora_b))
        {
            assert!(
                (left - right).abs() < 1.0e-5,
                "parallel result differs at layer {index}: {left} vs {right}"
            );
        }
        let (x, y) = &owned[index];
        assert!(cosine(&apply_lora(x, 30, parallel), y) > 0.99);
    }
}

fn temporary_directory(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "engram-rs-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn safetensors_and_peft_config_round_trip() {
    let layers = vec![
        CompiledLayer {
            lora_a: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            lora_b: vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            rank: 2,
            d_model: 4,
            target: "o_proj".to_string(),
            layer_index: 0,
            d_in: 4,
            d_out: 4,
        },
        CompiledLayer {
            lora_a: vec![-0.5; 8],
            lora_b: vec![0.25; 8],
            rank: 2,
            d_model: 4,
            target: "o_proj".to_string(),
            layer_index: 1,
            d_in: 4,
            d_out: 4,
        },
    ];
    let directory = temporary_directory("adapter-roundtrip");
    write_adapter(&directory, &layers, "test/base-model").unwrap();

    let bytes = std::fs::read(directory.join("adapter_model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(tensors.len(), 4);
    for (index, expected) in layers.iter().enumerate() {
        let prefix = format!("base_model.model.model.layers.{index}.self_attn.o_proj.lora_");
        let a = tensors.tensor(&format!("{prefix}A.weight")).unwrap();
        let b = tensors.tensor(&format!("{prefix}B.weight")).unwrap();
        assert_eq!(a.dtype(), Dtype::F32);
        assert_eq!(b.dtype(), Dtype::F32);
        assert_eq!(a.shape(), &[2, 4]);
        assert_eq!(b.shape(), &[4, 2]);
        let decode = |data: &[u8]| -> Vec<f32> {
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        assert_eq!(decode(a.data()), expected.lora_a);
        assert_eq!(decode(b.data()), expected.lora_b);
    }

    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("adapter_config.json")).unwrap())
            .unwrap();
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!(config["base_model_name_or_path"], "test/base-model");
    assert_eq!(config["r"], 2);
    assert_eq!(config["lora_alpha"], 2);
    assert_eq!(config["target_modules"], serde_json::json!(["o_proj"]));
    assert_eq!(config["task_type"], "CAUSAL_LM");
    assert_eq!(config["fan_in_fan_out"], false);

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn serializer_rejects_inconsistent_layers() {
    let layers = vec![
        CompiledLayer {
            lora_a: vec![0.0; 8],
            lora_b: vec![0.0; 8],
            rank: 2,
            d_model: 4,
            target: "o_proj".to_string(),
            layer_index: 0,
            d_in: 4,
            d_out: 4,
        },
        CompiledLayer {
            lora_a: vec![0.0; 3],
            lora_b: vec![0.0; 3],
            rank: 1,
            d_model: 3,
            target: "o_proj".to_string(),
            layer_index: 1,
            d_in: 3,
            d_out: 3,
        },
    ];
    let directory = temporary_directory("invalid-adapter");
    let error = write_adapter(&directory, &layers, "test/base-model").unwrap_err();
    assert!(error.contains("differ"));
}

#[cfg(not(debug_assertions))]
#[test]
fn benchmark_28_layers_d2048_rank16_release() {
    const LAYERS: usize = 28;
    const D: usize = 2048;
    const RANK: usize = 16;
    const N: usize = 32;

    let owned: Vec<_> = (0..LAYERS)
        .map(|index| low_rank_data(N, D, RANK, 9_000 + index as u64))
        .collect();
    let layers: Vec<_> = owned
        .iter()
        .map(|(x, y)| LayerData {
            hidden_states: x,
            attention_output: y,
            null_states: None,
            d_model: D,
        })
        .collect();
    let compiler_config = LayerCompilerConfig::new(RANK, N, 1.0e-3, 0.0);

    let start = std::time::Instant::now();
    let compiled = compile_all_layers(&layers, &compiler_config).unwrap();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "BENCHMARK compile_all_layers: layers={LAYERS} d_model={D} rank={RANK} wall_time_ms={elapsed_ms:.3} target_under_400ms={}",
        elapsed_ms < 400.0
    );
    assert!(
        elapsed_ms < 400.0,
        "28-layer compilation took {elapsed_ms:.3} ms, exceeding the 400 ms budget"
    );
    assert_eq!(compiled.len(), LAYERS);
    assert!(compiled.iter().all(|layer| {
        layer.rank == RANK
            && layer.d_model == D
            && layer.lora_a.len() == RANK * D
            && layer.lora_b.len() == D * RANK
            && layer.lora_a.iter().all(|value| value.is_finite())
            && layer.lora_b.iter().all(|value| value.is_finite())
    }));
}

#[test]
fn strict_null_space_projection_zeros_background_to_machine_precision() {
    let d = 16;
    let rank = 4;
    let n = 32;
    let null_rows = 8;
    let (x, y) = low_rank_data(n, d, rank, 12345);
    let mut rng = Lcg::new(67890);
    let null_states: Vec<f32> = (0..null_rows * d).map(|_| rng.next_f32()).collect();

    let layer_data = LayerData {
        hidden_states: &x,
        attention_output: &y,
        null_states: None,
        d_model: d,
    };
    let compiler_config = LayerCompilerConfig::new(rank, n, 1.0e-4, 0.0);
    let mut layer = compile_layer_weights(&layer_data, &compiler_config).unwrap();
    project_layer_null_space(&mut layer, &null_states).unwrap();

    // Adapter output on background null states must be 0.0000
    let null_output = apply_lora(&null_states, null_rows, &layer);
    let max_leakage = null_output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_leakage < 1.0e-6,
        "adapter output on background null states exceeds machine precision: {max_leakage}"
    );

    // Any normalized vector in span(null_states) must also produce 0.0000 output
    let mut comb = vec![0.0f32; d];
    for r in 0..null_rows {
        let weight = ((r as f32 + 1.0) * 0.3) / (null_rows as f32);
        for c in 0..d {
            comb[c] += weight * null_states[r * d + c];
        }
    }
    let comb_output = apply_lora(&comb, 1, &layer);
    let max_comb_leakage = comb_output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_comb_leakage < 1.0e-6,
        "adapter output on span(null_states) exceeds machine precision: {max_comb_leakage}"
    );
}

#[test]
fn mlp_down_proj_asymmetric_compilation_and_reconstruction() {
    let d_in = 16;
    let d_out = 8;
    let rank = 3;
    let n = 24;
    let mut rng = Lcg::new(42);

    let x: Vec<f32> = (0..n * d_in).map(|_| rng.next_f32()).collect();
    let a: Vec<f32> = (0..d_in * rank).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..d_out * rank).map(|_| rng.next_f32()).collect();
    let mut y = vec![0.0f32; n * d_out];
    for row in 0..n {
        for component in 0..rank {
            let mut proj = 0.0f32;
            for c in 0..d_in {
                proj += x[row * d_in + c] * a[c * rank + component];
            }
            for col in 0..d_out {
                y[row * d_out + col] += proj * b[col * rank + component];
            }
        }
    }

    let target = TargetData {
        target: "down_proj".to_string(),
        layer_index: 0,
        input: &x,
        output: &y,
        null_states: None,
        d_in,
        d_out,
    };
    let cfg = LayerCompilerConfig::new(rank, n, 1.0e-5, 0.0);
    let compiled = compile_target_weights(&target, &cfg).unwrap();

    assert_eq!(compiled.target, "down_proj");
    assert_eq!(compiled.d_in, d_in);
    assert_eq!(compiled.d_out, d_out);
    assert_eq!(compiled.rank, rank);
    assert_eq!(compiled.lora_a.len(), rank * d_in);
    assert_eq!(compiled.lora_b.len(), d_out * rank);

    // Verify reconstruction X A^T B^T ~= Y
    let mut recon = vec![0.0f32; n * d_out];
    for row in 0..n {
        let mut proj = vec![0.0f32; rank];
        for (comp, p) in proj.iter_mut().enumerate() {
            for c in 0..d_in {
                *p += x[row * d_in + c] * compiled.lora_a[comp * d_in + c];
            }
        }
        for col in 0..d_out {
            for (comp, p) in proj.iter().enumerate() {
                recon[row * d_out + col] += p * compiled.lora_b[col * rank + comp];
            }
        }
    }

    let sim = cosine(&recon, &y);
    assert!(
        sim > 0.999,
        "MLP down_proj reconstruction cosine similarity is too low: {sim}"
    );
}

#[test]
fn multi_target_spectral_energy_allocation_and_rank_pattern() {
    let d_model = 8;
    let intermediate_size = 16;
    let n = 20;

    let (o_x, o_y) = low_rank_data(n, d_model, 2, 101);
    let (mlp_x, mlp_y) = {
        let mut rng = Lcg::new(202);
        let x: Vec<f32> = (0..n * intermediate_size).map(|_| rng.next_f32()).collect();
        let a: Vec<f32> = (0..intermediate_size * 4).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..d_model * 4).map(|_| rng.next_f32()).collect();
        let mut y = vec![0.0f32; n * d_model];
        for row in 0..n {
            for comp in 0..4 {
                let mut p = 0.0f32;
                for c in 0..intermediate_size {
                    p += x[row * intermediate_size + c] * a[c * 4 + comp];
                }
                for col in 0..d_model {
                    y[row * d_model + col] += p * b[col * 4 + comp];
                }
            }
        }
        (x, y)
    };

    let targets = vec![
        TargetData {
            target: "o_proj".to_string(),
            layer_index: 0,
            input: &o_x,
            output: &o_y,
            null_states: None,
            d_in: d_model,
            d_out: d_model,
        },
        TargetData {
            target: "down_proj".to_string(),
            layer_index: 0,
            input: &mlp_x,
            output: &mlp_y,
            null_states: None,
            d_in: intermediate_size,
            d_out: d_model,
        },
    ];

    let mut cfg = LayerCompilerConfig::new(8, n, 1.0e-4, 0.0);
    cfg.energy_threshold = Some(0.95);
    cfg.min_rank = 1;
    cfg.max_rank = 8;

    let compiled = compile_all_targets(&targets, &cfg).unwrap();
    assert_eq!(compiled.len(), 2);
    let o_rank = compiled[0].rank;
    let mlp_rank = compiled[1].rank;
    assert_ne!(o_rank, mlp_rank, "ranks should vary across targets");
    let max_rank = o_rank.max(mlp_rank);

    let directory = temporary_directory("multi-target-rank-pattern");
    write_adapter(&directory, &compiled, "test/llama-model").unwrap();

    let config_bytes = std::fs::read(directory.join("adapter_config.json")).unwrap();
    let config: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();

    assert_eq!(
        config["target_modules"],
        serde_json::json!(["o_proj", "down_proj"])
    );
    assert_eq!(config["r"], serde_json::json!(max_rank));
    assert_eq!(
        config["rank_pattern"]["layers.0.self_attn.o_proj"],
        serde_json::json!(o_rank)
    );
    assert_eq!(
        config["rank_pattern"]["layers.0.mlp.down_proj"],
        serde_json::json!(mlp_rank)
    );

    let sf_bytes = std::fs::read(directory.join("adapter_model.safetensors")).unwrap();
    let sf = SafeTensors::deserialize(&sf_bytes).unwrap();
    let names: Vec<String> = sf.names().into_iter().map(|s| s.to_string()).collect();
    assert!(names
        .iter()
        .any(|n| n == "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"));
    assert!(names
        .iter()
        .any(|n| n == "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"));
    assert!(names
        .iter()
        .any(|n| n == "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"));
    assert!(names
        .iter()
        .any(|n| n == "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"));

    std::fs::remove_dir_all(directory).unwrap();
}
