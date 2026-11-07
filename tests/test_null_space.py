"""Tests for Direction 3: Strict Null-Space Orthogonal Projection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import (
    DEFAULT_CALIBRATION_PROMPTS,
    EngramCompiler,
    calibrate_background_anchors,
    compute_null_space_projector,
    get_family_background_anchors,
    project_lora_to_null_space,
    project_to_null_space,
)


class ByteTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        token_ids = [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        _name_or_path="tiny-llama-null",
    )
    return LlamaForCausalLM(config)


def test_null_space_projector_algebraic_properties():
    """Verify projector is symmetric, idempotent, and annihilates X_null."""
    rng = np.random.default_rng(42)
    k, d = 10, 32

    # Background representations matrix
    x_null = rng.standard_normal((k, d), dtype=np.float32)
    p_null = compute_null_space_projector(x_null)

    assert p_null.shape == (d, d)

    # 1. Symmetry: P^T = P
    np.testing.assert_allclose(p_null, p_null.T, atol=1e-7)

    # 2. Idempotence: P^2 = P
    p_sq = p_null @ p_null
    np.testing.assert_allclose(p_sq, p_null, atol=1e-6)

    # 3. Annihilation: X_null @ P_null = 0 to machine precision
    residual = x_null @ p_null
    max_res = np.max(np.abs(residual))
    assert max_res < 1e-6, f"X_null @ P did not annihilate: max {max_res}"

    # Also verify PyTorch implementation
    x_null_pt = torch.from_numpy(x_null)
    p_null_pt = compute_null_space_projector(x_null_pt)
    assert p_null_pt.shape == (d, d)
    np.testing.assert_allclose(p_null_pt.numpy(), p_null, atol=1e-6)
    residual_pt = torch.matmul(x_null_pt, p_null_pt)
    assert torch.max(torch.abs(residual_pt)).item() < 1e-6


def test_strict_null_projection_zeros_background_to_machine_precision():
    """Verify Delta W * (I - X^+ X) produces exactly 0.0000 on ANY background vector."""
    rng = np.random.default_rng(123)
    k, d = 8, 24
    x_null = rng.standard_normal((k, d), dtype=np.float32)

    # Arbitrary non-zero weight matrix (e.g. from regression)
    delta_w = rng.standard_normal((d, d), dtype=np.float32)
    # Parameter projection: side="right" ensures x @ (delta_w @ P)^T = x @ P @ delta_w^T = 0
    delta_w_proj = project_to_null_space(delta_w, x_null, side="right")

    # 1. Test on each row of X_null
    for i in range(k):
        x_bg = x_null[i : i + 1]
        out = x_bg @ delta_w_proj.T
        max_val = np.max(np.abs(out))
        assert max_val < 1e-6, f"Row {i} produced non-zero adapter response: {max_val}"

    # 2. Test on arbitrary linear combinations of background vectors
    for trial in range(20):
        coeffs = rng.standard_normal(k, dtype=np.float32)
        # Random vector in range(X_null)
        v_bg = coeffs @ x_null
        out = v_bg @ delta_w_proj.T
        max_val = np.max(np.abs(out))
        # Normalized response
        norm_v = np.linalg.norm(v_bg)
        if norm_v > 0:
            rel_response = max_val / norm_v
            assert rel_response < 1e-6, f"Linear combination produced response: {rel_response}"


def test_orthogonal_signals_are_preserved():
    """Verify signals orthogonal to X_null are not destroyed by null projection."""
    rng = np.random.default_rng(456)
    k, d = 5, 20
    x_null = rng.standard_normal((k, d), dtype=np.float32)

    # Find orthogonal complement basis
    q, _ = np.linalg.qr(x_null.T, mode="complete")
    # q[:, k:] are orthogonal to all rows of x_null
    v_ortho = q[:, k : k + 1].T.astype(np.float32)

    delta_w = rng.standard_normal((d, d), dtype=np.float32)
    delta_w_proj = project_to_null_space(delta_w, x_null, side="right")

    out_orig = v_ortho @ delta_w.T
    out_proj = v_ortho @ delta_w_proj.T

    # The projection should preserve the response on orthogonal directions
    np.testing.assert_allclose(out_proj, out_orig, rtol=1e-5, atol=1e-5)


def test_lora_factor_strict_null_projection():
    """Verify projecting lora_a directly zeroes out adapter response for any background vector."""
    rng = np.random.default_rng(789)
    k, d, rank = 6, 32, 4
    x_null = rng.standard_normal((k, d), dtype=np.float32)

    lora_a = rng.standard_normal((rank, d), dtype=np.float32)
    lora_b = rng.standard_normal((d, rank), dtype=np.float32)

    lora_a_proj, lora_b_out = project_lora_to_null_space(lora_a, lora_b, x_null)
    assert lora_b_out is lora_b
    assert lora_a_proj.shape == (rank, d)

    # In LoRA forward pass: delta_y = (x @ lora_a.T) @ lora_b.T
    # For any vector in range(X_null):
    for i in range(k):
        x_bg = x_null[i : i + 1]
        delta_y = (x_bg @ lora_a_proj.T) @ lora_b.T
        max_leakage = np.max(np.abs(delta_y))
        norm_scale = np.linalg.norm(x_bg)
        assert max_leakage / norm_scale < 1e-5, f"LoRA leakage on background token {i}: {max_leakage / norm_scale}"


def test_background_anchors_calibration_and_families():
    """Verify calibration helper and family anchor generator."""
    model = tiny_model()
    tokenizer = ByteTokenizer()

    # 1. Calibrate from model
    prompts = DEFAULT_CALIBRATION_PROMPTS[:3]
    anchors = calibrate_background_anchors(model, tokenizer, prompts=prompts, max_tokens=32)
    assert len(anchors) == 2  # 2 decoder layers
    assert all(arr.shape[-1] == 16 for arr in anchors.values())
    assert all(arr.shape[0] > 0 for arr in anchors.values())

    # 2. Family anchor generator
    for family in ["llama", "llama3", "mistral", "qwen", "generic"]:
        anchors_fam = get_family_background_anchors(family, d_model=64, num_anchors=16)
        assert anchors_fam.shape == (16, 64)
        # Check orthonormality
        gram = anchors_fam @ anchors_fam.T
        np.testing.assert_allclose(gram, np.eye(16), atol=1e-5)


def test_compiler_with_strict_null_projection(tmp_path: Path):
    """Test full EngramCompiler flow with strict null space projection enabled."""
    model = tiny_model()
    tokenizer = ByteTokenizer()
    compiler = EngramCompiler(model, tokenizer)

    rng = np.random.default_rng(101)
    # 2 layers, d_model=16
    null_states = [
        rng.standard_normal((8, 16), dtype=np.float32),
        rng.standard_normal((8, 16), dtype=np.float32),
    ]

    out_dir = compiler.compile_text(
        context_text="Confidential Project Alpha credentials 4821-XYZ.",
        output_dir=str(tmp_path / "null_adapter"),
        rank=2,
        null_states=null_states,
        strict_null_projection=True,
    )

    adapter_path = Path(out_dir)
    assert (adapter_path / "adapter_model.safetensors").is_file()

    # Verify adapter can be loaded with PEFT
    peft_model = EngramCompiler.inject_engram(model, str(adapter_path))
    assert peft_model is not None
