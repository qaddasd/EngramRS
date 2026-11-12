use engram_rs::math_core::{
    compile_all_layers, compile_all_targets, compile_layer_weights, compile_target_weights,
    project_layer_null_space, CompiledLayer, LayerCompilerConfig, LayerData, TargetData,
};
use engram_rs::serializer::{
    f16_bytes, module_sub_path, write_adapter, write_safetensors_dtype, WeightDtype,
};
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
fn safetensors_fp16_round_trip() {
    use engram_rs::serializer::write_adapter_typed;
    let layers = vec![CompiledLayer {
        lora_a: vec![0.5, -1.0, 2.0, -3.5],
        lora_b: vec![1.25, -0.75, 4.0, 0.0],
        rank: 2,
        d_model: 2,
        target: "o_proj".to_string(),
        layer_index: 0,
        d_in: 2,
        d_out: 2,
    }];
    let directory = temporary_directory("adapter-fp16-roundtrip");
    write_adapter_typed(&directory, &layers, "test/base-model", true).unwrap();

    let bytes = std::fs::read(directory.join("adapter_model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let a = tensors
        .tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight")
        .unwrap();
    assert_eq!(a.dtype(), Dtype::F16);
    assert_eq!(a.shape(), &[2, 2]);

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn serializer_rejects_inconsistent_layers() {
    // Mixed d_out across targets is legal now (q/k/v/o + gate/up/down adapters);
    // genuinely inconsistent layers (lora buffer length != rank * dim) must
    // still be rejected.
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
            lora_a: vec![0.0; 3], // rank * d_in would be 2 * 4 = 8
            lora_b: vec![0.0; 8],
            rank: 2,
            d_model: 4,
            target: "q_proj".to_string(),
            layer_index: 1,
            d_in: 4,
            d_out: 4,
        },
    ];
    let directory = temporary_directory("invalid-adapter");
    let error = write_adapter(&directory, &layers, "test/base-model").unwrap_err();
    assert!(error.contains("does not match"));
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

// ---------------------------------------------------------------------------
// Regression tests for the verified bug-fix batch (R1, R2, R4, R5, R6).
// ---------------------------------------------------------------------------

/// Reconstruct W = A^T B^T (shape [d_in, d_out]) from a compiled LoRA pair.
fn lora_to_weights(layer: &CompiledLayer) -> Vec<f32> {
    let mut w = vec![0.0f32; layer.d_in * layer.d_out];
    for i in 0..layer.d_in {
        for j in 0..layer.d_out {
            let mut acc = 0.0f32;
            for r in 0..layer.rank {
                acc += layer.lora_a[r * layer.d_in + i] * layer.lora_b[j * layer.rank + r];
            }
            w[i * layer.d_out + j] = acc;
        }
    }
    w
}

fn rect_low_rank_data(
    n: usize,
    d_in: usize,
    d_out: usize,
    rank: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let x: Vec<f32> = (0..n * d_in).map(|_| rng.next_f32()).collect();
    let a: Vec<f32> = (0..d_in * rank).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..d_out * rank).map(|_| rng.next_f32()).collect();
    let mut y = vec![0.0f32; n * d_out];
    for row in 0..n {
        for component in 0..rank {
            let mut projected = 0.0f32;
            for inner in 0..d_in {
                projected += x[row * d_in + inner] * a[inner * rank + component];
            }
            for col in 0..d_out {
                y[row * d_out + col] += projected * b[col * rank + component];
            }
        }
    }
    (x, y)
}

/// R1: relative ridge must make the solution invariant to uniform rescaling
/// of (X, Y), and primal/dual must stay numerically equivalent.
#[test]
fn adaptive_ridge_scale_invariance_and_primal_dual_equivalence() {
    let n = 48;
    let d = 8;
    let mut rng = Lcg::new(555);
    // Ill-conditioned X: column scales 10 .. 0.01 (cond(X^T X) ~ 1e6).
    let mut x = vec![0.0f32; n * d];
    for row in 0..n {
        for col in 0..d {
            let scale = 10f32.powf(1.0 - 3.0 * col as f32 / (d - 1) as f32);
            x[row * d + col] = rng.next_f32() * scale;
        }
    }
    let w_true: Vec<f32> = (0..d * d).map(|_| rng.next_f32()).collect();
    let mut y = vec![0.0f32; n * d];
    for row in 0..n {
        for col in 0..d {
            let mut acc = 0.0f32;
            for k in 0..d {
                acc += x[row * d + k] * w_true[k * d + col];
            }
            y[row * d + col] = acc + 0.01 * rng.next_f32();
        }
    }

    let mut cfg = config(d, n);
    cfg.lambda_reg = 0.1; // relative epsilon: lambda_eff = 0.1 * trace(gram) / d_in
    let base = compile_layer_weights(
        &LayerData {
            hidden_states: &x,
            attention_output: &y,
            null_states: None,
            d_model: d,
        },
        &cfg,
    )
    .unwrap();
    let xs: Vec<f32> = x.iter().map(|v| v * 1000.0).collect();
    let ys: Vec<f32> = y.iter().map(|v| v * 1000.0).collect();
    let scaled = compile_layer_weights(
        &LayerData {
            hidden_states: &xs,
            attention_output: &ys,
            null_states: None,
            d_model: d,
        },
        &cfg,
    )
    .unwrap();
    let w_base = lora_to_weights(&base);
    let w_scaled = lora_to_weights(&scaled);
    let sim = cosine(&w_base, &w_scaled);
    assert!(
        sim > 0.999,
        "relative ridge broke scale invariance: cosine={sim}"
    );

    // Primal vs dual on identical anchored data (n < d_in forces the dual
    // formulation in production; here both are called explicitly).
    let n2 = 6;
    let d2 = 10;
    let (x2, y2) = rect_low_rank_data(n2, d2, d2, 4, 777);
    let anchored = engram_rs::math_core::AnchoredMatrices {
        x: faer::Mat::from_fn(n2, d2, |r, c| x2[r * d2 + c]),
        y: faer::Mat::from_fn(n2, d2, |r, c| y2[r * d2 + c]),
    };
    let mut cfg2 = config(6, n2);
    cfg2.lambda_reg = 1e-3;
    let primal =
        engram_rs::math_core::solve_primal_and_factorize(&anchored, None, 0, &cfg2, "o_proj", 0)
            .unwrap();
    let dual =
        engram_rs::math_core::solve_dual_and_factorize(&anchored, None, 0, &cfg2, "o_proj", 0)
            .unwrap();
    let w_primal = lora_to_weights(&primal);
    let w_dual = lora_to_weights(&dual);
    let equivalence = cosine(&w_primal, &w_dual);
    assert!(
        equivalence > 0.999,
        "primal/dual equivalence broken: cosine={equivalence}"
    );
}

/// R2: dense streaming solve must run Cholesky in f64 and match a direct f64
/// reference on ill-conditioned data (cond(X^T X) ~ 1e8) far better than f32.
#[test]
fn dense_streaming_f64_solve_beats_f32_on_ill_conditioned_data() {
    use engram_rs::streaming::CovarianceAccumulator;
    use faer::linalg::solvers::{Cholesky, SpSolver};
    use faer::{Mat, Side};

    let d_in = 6usize;
    let d_out = 4usize;
    let n = 64usize;
    let mut rng = Lcg::new(2024);
    let mut x = vec![0.0f32; n * d_in];
    for row in 0..n {
        for col in 0..d_in {
            let scale = 10f32.powf(4.0 * (1.0 - col as f32 / (d_in - 1) as f32));
            x[row * d_in + col] = rng.next_f32() * scale;
        }
    }
    let w_true: Vec<f32> = (0..d_in * d_out).map(|_| rng.next_f32()).collect();
    let mut y = vec![0.0f32; n * d_out];
    for row in 0..n {
        for col in 0..d_out {
            let mut acc = 0.0f64;
            for k in 0..d_in {
                acc += f64::from(x[row * d_in + k]) * f64::from(w_true[k * d_out + col]);
            }
            y[row * d_out + col] = acc as f32;
        }
    }

    let lambda = 1e-4f32;
    let mut acc = CovarianceAccumulator::new(d_in, d_out);
    let chunk = 16;
    for c in 0..n / chunk {
        acc.add_chunk(
            &x[c * chunk * d_in..(c + 1) * chunk * d_in],
            &y[c * chunk * d_out..(c + 1) * chunk * d_out],
            chunk,
        )
        .unwrap();
    }
    let cfg = LayerCompilerConfig::new(d_out, 256, lambda, 0.0);
    let layer = acc.solve(&cfg, "o_proj", 0).unwrap();
    let w_est = lora_to_weights(&layer);

    // Direct f64 reference solve of (X^T X + lambda I) W = X^T Y.
    let mut gram = vec![0.0f64; d_in * d_in];
    let mut cross = vec![0.0f64; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_in {
            let mut s = 0.0f64;
            for r in 0..n {
                s += f64::from(x[r * d_in + i]) * f64::from(x[r * d_in + j]);
            }
            gram[i * d_in + j] = s;
        }
        gram[i * d_in + i] += f64::from(lambda);
        for j in 0..d_out {
            let mut s = 0.0f64;
            for r in 0..n {
                s += f64::from(x[r * d_in + i]) * f64::from(y[r * d_out + j]);
            }
            cross[i * d_out + j] = s;
        }
    }
    let gram_mat = Mat::from_fn(d_in, d_in, |i, j| gram[i * d_in + j]);
    let cross_mat = Mat::from_fn(d_in, d_out, |i, j| cross[i * d_out + j]);
    let w_ref = Cholesky::try_new(gram_mat.as_ref(), Side::Lower)
        .unwrap()
        .solve(cross_mat.as_ref());
    let rel_err = |w: &[f32]| -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..d_in {
            for j in 0..d_out {
                let reference = w_ref.read(i, j);
                let diff = f64::from(w[i * d_out + j]) - reference;
                num += diff * diff;
                den += reference * reference;
            }
        }
        (num / den).sqrt()
    };
    let err_f64 = rel_err(&w_est);
    assert!(
        err_f64 < 1e-4,
        "f64 streaming solve relative error too large: {err_f64}"
    );

    // Legacy f32 path (cast gram/cross to f32 before Cholesky) for comparison.
    let gram_f32 = Mat::from_fn(d_in, d_in, |i, j| gram[i * d_in + j] as f32);
    let cross_f32 = Mat::from_fn(d_in, d_out, |i, j| cross[i * d_out + j] as f32);
    let err_f32 = match Cholesky::try_new(gram_f32.as_ref(), Side::Lower) {
        Ok(cholesky) => {
            let w = cholesky.solve(cross_f32.as_ref());
            let flat: Vec<f32> = (0..d_in * d_out)
                .map(|idx| w.read(idx / d_out, idx % d_out))
                .collect();
            rel_err(&flat)
        }
        Err(_) => f64::INFINITY,
    };
    assert!(
        err_f64 * 10.0 < err_f32,
        "f64 solve ({err_f64}) must be much better than f32 ({err_f32})"
    );
}

/// Build A = U diag(s) V^T with deterministic pseudo-random orthogonal factors.
fn matrix_with_spectrum(n: usize, d: usize, singular_values: &[f32], seed: u64) -> faer::Mat<f32> {
    use faer::linalg::solvers::Qr;
    let mut rng = Lcg::new(seed);
    let r = singular_values.len().min(n).min(d);
    let u_rand = faer::Mat::from_fn(n, r, |_, _| rng.next_f32());
    let v_rand = faer::Mat::from_fn(d, r, |_, _| rng.next_f32());
    let u = Qr::new(u_rand.as_ref()).compute_thin_q();
    let v = Qr::new(v_rand.as_ref()).compute_thin_q();
    faer::Mat::from_fn(n, d, |i, j| {
        let mut acc = 0.0f32;
        for (k, &sigma) in singular_values.iter().enumerate().take(r) {
            acc += u.read(i, k) * sigma * v.read(j, k);
        }
        acc
    })
}

/// Replica of the OLD no-power, uniform-LCG randomized projection, used to
/// prove the new power-iteration rSVD strictly improves captured energy.
fn legacy_no_power_anchor_energy(x: &faer::Mat<f32>, k: usize) -> f64 {
    use faer::linalg::solvers::Qr;
    let width = (k + 8).min(x.nrows()).min(x.ncols());
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let omega = faer::Mat::from_fn(x.ncols(), width, |_, _| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 40) as f32 / ((1_u64 << 24) - 1) as f32;
        2.0 * unit - 1.0
    });
    let sampled = x.as_ref() * omega.as_ref();
    let q = Qr::new(sampled.as_ref()).compute_thin_q();
    let reduced = q.transpose() * x.as_ref();
    let svd = reduced.as_ref().thin_svd();
    let s = svd.s_diagonal();
    (0..k)
        .map(|i| f64::from(s.read(i)) * f64::from(s.read(i)))
        .sum()
}

/// R4: power-iteration rSVD must capture >= 95% of the full-SVD top-k energy
/// on a slowly-decaying spectrum and beat the old single-pass method.
#[test]
fn power_iteration_rsvd_captures_slowly_decaying_spectrum() {
    let n = 96;
    let d = 64;
    let k = 16;
    let s: Vec<f32> = (1..=d).map(|i| 1.0 / (i as f32).powf(0.3)).collect();
    let x = matrix_with_spectrum(n, d, &s, 31337);
    let y = faer::Mat::from_fn(n, 8, |r, c| ((r * 3 + c * 7 + 1) as f32).sin());

    let anchored = engram_rs::math_core::randomized_svd_anchors(&x, &y, k).unwrap();
    assert_eq!(anchored.x.nrows(), k);
    let mut captured = 0.0f64;
    for r in 0..anchored.x.nrows() {
        for c in 0..anchored.x.ncols() {
            let v = f64::from(anchored.x.read(r, c));
            captured += v * v;
        }
    }
    let full_top_k: f64 = s[..k].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    assert!(
        captured >= 0.95 * full_top_k,
        "power-iteration rSVD captured {captured} of {full_top_k} ({}%)",
        100.0 * captured / full_top_k
    );
    let legacy = legacy_no_power_anchor_energy(&x, k);
    assert!(
        captured > legacy,
        "power iterations must beat the old no-power method: {captured} vs {legacy}"
    );
}

/// R4: when k >= rank(A), randomized anchors must preserve the normal
/// equations (gram and cross) of the original data.
#[test]
fn randomized_anchors_preserve_gram_when_k_covers_rank() {
    let n = 64;
    let d = 32;
    let k = 16; // k >= rank = 8, k < min(n, d) so the randomized path is used
    let s = [10.0f32, 5.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.125];
    let x = matrix_with_spectrum(n, d, &s, 4242);
    let w = faer::Mat::from_fn(d, 12, |r, c| ((r * 5 + c * 3 + 2) as f32).cos() * 0.5);
    let y = x.as_ref() * w.as_ref();

    let anchored = engram_rs::math_core::randomized_svd_anchors(&x, &y, k).unwrap();
    let gram_ref = x.transpose() * x.as_ref();
    let gram_act = anchored.x.transpose() * anchored.x.as_ref();
    let cross_ref = x.transpose() * y.as_ref();
    let cross_act = anchored.x.transpose() * anchored.y.as_ref();

    let rel_fro = |a: faer::MatRef<'_, f32>, b: faer::MatRef<'_, f32>| -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                let diff = f64::from(a.read(r, c)) - f64::from(b.read(r, c));
                let reference = f64::from(b.read(r, c));
                num += diff * diff;
                den += reference * reference;
            }
        }
        (num / den).sqrt()
    };
    let gram_err = rel_fro(gram_act.as_ref(), gram_ref.as_ref());
    let cross_err = rel_fro(cross_act.as_ref(), cross_ref.as_ref());
    assert!(gram_err < 1e-4, "gram relative error {gram_err}");
    assert!(cross_err < 1e-4, "cross relative error {cross_err}");
}

/// R5: one adapter may mix targets with different d_out (o_proj 64,
/// down_proj 128, q_proj 32); PEFT keys and target_modules must be correct.
#[test]
fn mixed_shape_multi_target_adapter_round_trip() {
    let n = 48;
    let rank = 4;
    let (o_x, o_y) = rect_low_rank_data(n, 64, 64, rank, 11);
    let (dn_x, dn_y) = rect_low_rank_data(n, 256, 128, rank, 22);
    let (q_x, q_y) = rect_low_rank_data(n, 32, 32, rank, 33);
    let targets = vec![
        TargetData {
            target: "o_proj".to_string(),
            layer_index: 0,
            input: &o_x,
            output: &o_y,
            null_states: None,
            d_in: 64,
            d_out: 64,
        },
        TargetData {
            target: "down_proj".to_string(),
            layer_index: 0,
            input: &dn_x,
            output: &dn_y,
            null_states: None,
            d_in: 256,
            d_out: 128,
        },
        TargetData {
            target: "q_proj".to_string(),
            layer_index: 0,
            input: &q_x,
            output: &q_y,
            null_states: None,
            d_in: 32,
            d_out: 32,
        },
    ];
    let cfg = LayerCompilerConfig::new(rank, n, 1e-4, 0.0);
    let compiled = compile_all_targets(&targets, &cfg).unwrap();
    assert_eq!(compiled.len(), 3);

    let directory = temporary_directory("mixed-shape-adapter");
    write_adapter(&directory, &compiled, "test/mixed-model").unwrap();

    let bytes = std::fs::read(directory.join("adapter_model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(tensors.len(), 6);
    let decode = |data: &[u8]| -> Vec<f32> {
        data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    };
    let expectations = [
        ("self_attn.o_proj", 64usize, 64usize, &compiled[0]),
        ("mlp.down_proj", 256, 128, &compiled[1]),
        ("self_attn.q_proj", 32, 32, &compiled[2]),
    ];
    for (sub_path, d_in, d_out, layer) in expectations {
        let prefix = format!("base_model.model.model.layers.0.{sub_path}.lora_");
        let a = tensors.tensor(&format!("{prefix}A.weight")).unwrap();
        let b = tensors.tensor(&format!("{prefix}B.weight")).unwrap();
        assert_eq!(a.dtype(), Dtype::F32);
        assert_eq!(a.shape(), &[rank, d_in]);
        assert_eq!(b.shape(), &[d_out, rank]);
        assert_eq!(decode(a.data()), layer.lora_a);
        assert_eq!(decode(b.data()), layer.lora_b);
    }

    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("adapter_config.json")).unwrap())
            .unwrap();
    let target_modules = config["target_modules"].as_array().unwrap();
    for name in ["o_proj", "down_proj", "q_proj"] {
        assert!(
            target_modules.iter().any(|v| v == name),
            "target_modules missing {name}: {target_modules:?}"
        );
    }
    assert_eq!(config["r"], serde_json::json!(rank));

    // Remaining attention / MLP module mappings.
    assert_eq!(module_sub_path("k_proj"), "self_attn.k_proj");
    assert_eq!(module_sub_path("v_proj"), "self_attn.v_proj");
    assert_eq!(module_sub_path("gate_proj"), "mlp.gate_proj");
    assert_eq!(module_sub_path("up_proj"), "mlp.up_proj");
    assert_eq!(module_sub_path("o_proj"), "self_attn.o_proj");
    assert_eq!(module_sub_path("down_proj"), "mlp.down_proj");

    std::fs::remove_dir_all(directory).unwrap();
}

/// R6: f16 conversion must reject out-of-range values instead of writing inf.
#[test]
fn f16_overflow_returns_error() {
    let err = f16_bytes(&[1.0, 70000.0]).unwrap_err();
    assert!(
        err.contains("70000"),
        "error must name the offending value: {err}"
    );
    assert!(f16_bytes(&[f32::INFINITY]).is_err());
    assert!(f16_bytes(&[f32::NAN]).is_err());
    assert!(f16_bytes(&[65504.0, -65504.0, 0.0]).is_ok());

    let layers = vec![CompiledLayer {
        lora_a: vec![100000.0, 1.0, 2.0, 3.0],
        lora_b: vec![0.0; 4],
        rank: 2,
        d_model: 2,
        target: "o_proj".to_string(),
        layer_index: 0,
        d_in: 2,
        d_out: 2,
    }];
    let directory = temporary_directory("f16-overflow");
    let err =
        engram_rs::serializer::write_safetensors_typed(&directory, &layers, true).unwrap_err();
    assert!(
        err.contains("100000"),
        "writer must propagate the overflow: {err}"
    );
    if directory.exists() {
        std::fs::remove_dir_all(directory).unwrap();
    }
}

/// R6: BF16 output round-trips within bf16 precision (8 significand bits).
#[test]
fn bf16_round_trip_preserves_values() {
    let lora_a = vec![
        0.5f32,
        -1.25,
        std::f32::consts::PI,
        100.0,
        0.001,
        -0.0,
        1234.5,
        -7.75,
    ];
    let lora_b = vec![2.5f32, -0.125, 63.5, 0.0078125];
    let layers = vec![CompiledLayer {
        lora_a: lora_a.clone(),
        lora_b: lora_b.clone(),
        rank: 2,
        d_model: 2,
        target: "o_proj".to_string(),
        layer_index: 0,
        d_in: 4,
        d_out: 2,
    }];
    let directory = temporary_directory("bf16-roundtrip");
    std::fs::create_dir_all(&directory).unwrap();
    write_safetensors_dtype(&directory, &layers, WeightDtype::BF16).unwrap();

    let bytes = std::fs::read(directory.join("adapter_model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let prefix = "base_model.model.model.layers.0.self_attn.o_proj.lora_";
    let a = tensors.tensor(&format!("{prefix}A.weight")).unwrap();
    let b = tensors.tensor(&format!("{prefix}B.weight")).unwrap();
    assert_eq!(a.dtype(), Dtype::BF16);
    assert_eq!(b.dtype(), Dtype::BF16);
    assert_eq!(a.shape(), &[2, 4]);
    assert_eq!(b.shape(), &[2, 2]);

    let decode_bf16 = |data: &[u8]| -> Vec<f32> {
        data.chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32())
            .collect()
    };
    // bf16 round-to-nearest relative error is bounded by 2^-8 = 1/256.
    for (actual, expected) in decode_bf16(a.data()).iter().zip(&lora_a) {
        assert!(
            (actual - expected).abs() <= expected.abs() / 256.0 + 1e-9,
            "bf16 lora_A mismatch: {actual} vs {expected}"
        );
    }
    for (actual, expected) in decode_bf16(b.data()).iter().zip(&lora_b) {
        assert!(
            (actual - expected).abs() <= expected.abs() / 256.0 + 1e-9,
            "bf16 lora_B mismatch: {actual} vs {expected}"
        );
    }

    std::fs::remove_dir_all(directory).unwrap();
}
