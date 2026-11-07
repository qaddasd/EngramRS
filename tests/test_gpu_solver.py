from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import (
    GPULayerCompiler,
    build_peft_state_dict,
    compile_all_layers_to_safetensors,
    solve_all_layers_gpu,
    solve_layer_gpu,
)
from safetensors import safe_open


def test_solve_layer_gpu_matches_rust_primal(tmp_path: Path):
    """Verify that PyTorch GPU solver yields exact mathematical equivalence with Rust primal solver."""
    d = 16
    n = 32
    r = 4
    rng = np.random.default_rng(42)
    x_np = rng.normal(size=(n, d)).astype(np.float32)
    proj = rng.normal(size=(d, d)).astype(np.float32) / d
    y_np = x_np @ proj

    # Rust reference
    rust_dir = tmp_path / "rust_primal"
    rust_dir.mkdir(parents=True, exist_ok=True)
    compile_all_layers_to_safetensors(
        [x_np],
        [y_np],
        output_dir=rust_dir,
        rank=r,
        lambda_reg=1e-3,
        lambda_null=0.0,
        base_model_name_or_path="test-ref",
    )
    with safe_open(rust_dir / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        a_rust = f.get_tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight")
        b_rust = f.get_tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight")

    # PyTorch solver
    x_pt = torch.from_numpy(x_np)
    y_pt = torch.from_numpy(y_np)
    a_pt, b_pt = solve_layer_gpu(
        x_pt,
        y_pt,
        rank=r,
        lambda_reg=1e-3,
        lambda_null=0.0,
    )

    assert a_pt.shape == (r, d)
    assert b_pt.shape == (d, r)

    # Reconstructed linear map W = A^T B^T
    w_rec_rust = (a_rust.T @ b_rust.T).numpy()
    w_rec_pt = (a_pt.T @ b_pt.T).numpy()

    diff = np.max(np.abs(w_rec_rust - w_rec_pt))
    assert diff < 1e-4, f"PyTorch primal solver differs from Rust reference by {diff}"


def test_solve_layer_gpu_matches_rust_dual(tmp_path: Path):
    """Verify dual QR-accelerated solve when N < D matches Rust dual solver."""
    d = 32
    n = 8
    r = 4
    rng = np.random.default_rng(123)
    x_np = rng.normal(size=(n, d)).astype(np.float32)
    proj = rng.normal(size=(d, d)).astype(np.float32) / d
    y_np = x_np @ proj

    # Rust reference
    rust_dir = tmp_path / "rust_dual"
    rust_dir.mkdir(parents=True, exist_ok=True)
    compile_all_layers_to_safetensors(
        [x_np],
        [y_np],
        output_dir=rust_dir,
        rank=r,
        lambda_reg=1e-3,
        lambda_null=0.0,
        base_model_name_or_path="test-dual",
    )
    with safe_open(rust_dir / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        a_rust = f.get_tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight")
        b_rust = f.get_tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight")

    # PyTorch solver
    x_pt = torch.from_numpy(x_np)
    y_pt = torch.from_numpy(y_np)
    a_pt, b_pt = solve_layer_gpu(
        x_pt,
        y_pt,
        rank=r,
        lambda_reg=1e-3,
        lambda_null=0.0,
    )

    w_rec_rust = (a_rust.T @ b_rust.T).numpy()
    w_rec_pt = (a_pt.T @ b_pt.T).numpy()

    diff = np.max(np.abs(w_rec_rust - w_rec_pt))
    assert diff < 1e-4, f"PyTorch dual solver differs from Rust reference by {diff}"


def test_solve_layer_gpu_with_null_states(tmp_path: Path):
    """Verify solve with null states regularizing background activations toward zero."""
    d = 16
    n = 12
    null_n = 6
    r = 2
    rng = np.random.default_rng(99)
    x_np = rng.normal(size=(n, d)).astype(np.float32)
    null_np = rng.normal(size=(null_n, d)).astype(np.float32)
    proj = rng.normal(size=(d, d)).astype(np.float32) / d
    y_np = x_np @ proj

    rust_dir = tmp_path / "rust_null"
    rust_dir.mkdir(parents=True, exist_ok=True)
    compile_all_layers_to_safetensors(
        [x_np],
        [y_np],
        null_states=[null_np],
        output_dir=rust_dir,
        rank=r,
        lambda_reg=1e-3,
        lambda_null=0.5,
        base_model_name_or_path="test-null",
    )
    with safe_open(rust_dir / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        a_rust = f.get_tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight")
        b_rust = f.get_tensor("base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight")

    x_pt = torch.from_numpy(x_np)
    y_pt = torch.from_numpy(y_np)
    null_pt = torch.from_numpy(null_np)
    a_pt, b_pt = solve_layer_gpu(
        x_pt,
        y_pt,
        rank=r,
        lambda_reg=1e-3,
        lambda_null=0.5,
        null_states=null_pt,
    )

    w_rec_rust = (a_rust.T @ b_rust.T).numpy()
    w_rec_pt = (a_pt.T @ b_pt.T).numpy()

    diff = np.max(np.abs(w_rec_rust - w_rec_pt))
    assert diff < 1e-4, f"Solver with null states differs from Rust reference by {diff}"


def test_solve_layer_gpu_cuda_execution():
    """Verify GPU solver runs on CUDA and completes in low latency (<50ms)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    d = 256
    n = 64
    r = 8
    x_cuda = torch.randn(n, d, device="cuda")
    y_cuda = torch.randn(n, d, device="cuda")

    # Warmup CUDA context and cuSOLVER library handles
    _ = solve_layer_gpu(x_cuda, y_cuda, rank=r)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    a_cuda, b_cuda = solve_layer_gpu(x_cuda, y_cuda, rank=r)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    assert a_cuda.is_cuda
    assert b_cuda.is_cuda
    assert a_cuda.shape == (r, d)
    assert b_cuda.shape == (d, r)
    assert torch.isfinite(a_cuda).all()
    assert torch.isfinite(b_cuda).all()
    assert elapsed_ms < 50.0, f"CUDA solve latency was unexpectedly high: {elapsed_ms:.2f} ms"


def test_solve_layer_gpu_anchor_compression():
    """Verify max_anchors compresses rows via SVD anchors."""
    d = 32
    n = 100
    r = 4
    x = torch.randn(n, d)
    y = torch.randn(n, d)

    a, b = solve_layer_gpu(x, y, rank=r, max_anchors=16)
    assert a.shape == (r, d)
    assert b.shape == (d, r)
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()


def test_solve_layer_gpu_input_validation():
    """Verify thorough parameter and tensor validation."""
    x = torch.randn(10, 8)
    y = torch.randn(10, 8)

    # rank > d_model
    with pytest.raises(ValueError, match=r"rank .* cannot exceed d_model"):
        solve_layer_gpu(x, y, rank=10)

    # rank <= 0
    with pytest.raises(ValueError, match=r"rank must be a positive integer"):
        solve_layer_gpu(x, y, rank=0)

    # lambda_reg <= 0
    with pytest.raises(ValueError, match=r"lambda_reg must be finite and positive"):
        solve_layer_gpu(x, y, rank=4, lambda_reg=0.0)

    # lambda_null < 0
    with pytest.raises(ValueError, match=r"lambda_null must be finite and non-negative"):
        solve_layer_gpu(x, y, rank=4, lambda_null=-0.1)

    # Shape mismatch
    with pytest.raises(ValueError, match=r"shape .* does not match"):
        solve_layer_gpu(x, torch.randn(10, 16), rank=4)

    # Non-finite values
    x_nan = x.clone()
    x_nan[0, 0] = float("nan")
    with pytest.raises(ValueError, match=r"non-finite"):
        solve_layer_gpu(x_nan, y, rank=4)


def test_solve_all_layers_and_peft_state_dict():
    """Verify multi-layer solving and PEFT state dict formatting."""
    num_layers = 3
    d = 16
    n = 20
    r = 4

    xs = [torch.randn(n, d) for _ in range(num_layers)]
    ys = [torch.randn(n, d) for _ in range(num_layers)]

    weights = solve_all_layers_gpu(xs, ys, rank=r)
    assert len(weights) == num_layers

    state_dict = build_peft_state_dict(weights, target_module="self_attn.o_proj")
    assert len(state_dict) == num_layers * 2

    for l in range(num_layers):
        a_key = f"base_model.model.model.layers.{l}.self_attn.o_proj.lora_A.weight"
        b_key = f"base_model.model.model.layers.{l}.self_attn.o_proj.lora_B.weight"
        assert a_key in state_dict
        assert b_key in state_dict
        assert state_dict[a_key].shape == (r, d)
        assert state_dict[b_key].shape == (d, r)


def test_gpu_layer_compiler_interface():
    """Verify GPULayerCompiler helper interface."""
    compiler = GPULayerCompiler(rank=2)
    xs = [torch.randn(10, 8), torch.randn(10, 8)]
    ys = [torch.randn(10, 8), torch.randn(10, 8)]

    state_dict = compiler.compile_activations(xs, ys)
    assert len(state_dict) == 4
    for key, tensor in state_dict.items():
        assert torch.isfinite(tensor).all()
