"""Tests for Direction 2: Streaming Covariance Accumulation & Fast Randomized SVD."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import (
    EngramCompiler,
    StreamingCovarianceAccumulator,
    randomized_svd,
)


class ByteTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
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
        _name_or_path="tiny-llama-streaming",
    )
    return LlamaForCausalLM(config)


def test_streaming_covariance_equivalence_to_full_batch():
    """Verify streaming accumulation gives identical results to full-batch regression."""
    rng = np.random.default_rng(42)
    n = 2048
    d = 32
    rank = 8
    lambda_reg = 1e-4

    # Full data
    x_full = rng.standard_normal((n, d), dtype=np.float32)
    true_weights = rng.standard_normal((d, d), dtype=np.float32) / np.sqrt(d)
    y_full = x_full @ true_weights + 0.01 * rng.standard_normal((n, d), dtype=np.float32)

    # Full-batch normal equations
    c_full = x_full.T.astype(np.float64) @ x_full.astype(np.float64)
    p_full = x_full.T.astype(np.float64) @ y_full.astype(np.float64)
    w_full = np.linalg.solve(c_full + lambda_reg * np.eye(d, dtype=np.float64), p_full).astype(np.float32)

    # Streaming accumulator with small chunks
    accumulator = StreamingCovarianceAccumulator(d_model=d, layer_indices=[0], use_double_precision=True)
    chunk_size = 256
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        accumulator.add_chunk(0, x_full[start:end], y_full[start:end])

    # Check accumulated covariance matrices
    c_stream = accumulator._c[0]
    p_stream = accumulator._p[0]
    np.testing.assert_allclose(c_stream, c_full, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(p_stream, p_full, rtol=1e-5, atol=1e-5)

    # Check solved weights
    solved = accumulator.solve(lambda_reg=lambda_reg, rank=rank, method="svd")
    lora_a, lora_b = solved[0]
    w_stream = (lora_a.T @ lora_b.T).astype(np.float32)

    # Low-rank approximation of w_full
    u_f, s_f, vh_f = np.linalg.svd(w_full, full_matrices=False)
    w_full_rank_k = (u_f[:, :rank] * s_f[:rank]) @ vh_f[:rank, :]

    # Equivalence: stream reconstructed low-rank map should match full-batch low-rank map
    np.testing.assert_allclose(w_stream, w_full_rank_k, rtol=1e-4, atol=1e-4)
    cos_sim = np.sum(w_stream * w_full_rank_k) / (np.linalg.norm(w_stream) * np.linalg.norm(w_full_rank_k))
    assert cos_sim > 0.99999, f"Cosine similarity too low: {cos_sim}"


def test_streaming_memory_remains_strictly_bounded_for_100k_tokens():
    """Verify memory is strictly O(d^2) and stays constant as token count scales to 100k+."""
    d = 64
    accumulator = StreamingCovarianceAccumulator(d_model=d, layer_indices=[0, 1], use_double_precision=True)

    # Theoretical size: 2 matrices * d^2 * 8 bytes (float64) per layer
    expected_layer_bytes = 2 * (d * d) * 8
    expected_total_bytes = 2 * expected_layer_bytes

    rng = np.random.default_rng(123)
    chunk_tokens = 1000
    total_chunks = 100  # 100,000 tokens total!

    memories = []
    for _ in range(total_chunks):
        x_chunk = rng.standard_normal((chunk_tokens, d), dtype=np.float32)
        y_chunk = rng.standard_normal((chunk_tokens, d), dtype=np.float32)
        accumulator.add_chunk(0, x_chunk, y_chunk)
        accumulator.add_chunk(1, x_chunk, y_chunk)
        memories.append(accumulator.memory_bytes())

    # Total tokens processed: 100,000
    assert accumulator.total_tokens == 100_000
    assert accumulator.memory_bytes() == expected_total_bytes

    # Memory must be completely flat across all 100 iterations (O(d^2), not O(N*d))
    assert len(set(memories)) == 1, "Memory grew with token count!"
    # Compare with what full O(N * d) would have cost: 100,000 * 64 * 4 * 2 layers = ~51.2 MB
    naive_activation_bytes = 100_000 * d * 4 * 2
    assert accumulator.memory_bytes() < naive_activation_bytes / 50


def test_rust_streaming_accumulator_32k_tokens():
    """Verify RustStreamingAccumulator handles 32,000 tokens incrementally without memory growth."""
    try:
        from engram_rs.engram_rs import RustStreamingAccumulator
    except ImportError:
        pytest.skip("Rust native extension not built")

    d = 32
    acc = RustStreamingAccumulator(d, d)
    rng = np.random.default_rng(42)

    chunk_size = 1000
    total_chunks = 32
    total_tokens = total_chunks * chunk_size  # 32,000 tokens

    for _ in range(total_chunks):
        x = rng.standard_normal((chunk_size, d), dtype=np.float32)
        y = rng.standard_normal((chunk_size, d), dtype=np.float32)
        acc.add_chunk(x, y)

    assert acc.total_tokens == total_tokens
    lora_a, lora_b = acc.solve(rank=8, lambda_reg=1e-4)
    assert lora_a.shape == (8, d)
    assert lora_b.shape == (d, 8)
    assert np.isfinite(lora_a).all()
    assert np.isfinite(lora_b).all()


def test_randomized_svd_halko_properties():
    """Test fast randomized SVD with power iterations against deterministic SVD."""
    rng = np.random.default_rng(999)
    m, n, k = 128, 96, 12

    # Construct matrix with rapidly decaying singular values
    u_true, _ = np.linalg.qr(rng.standard_normal((m, k)))
    v_true, _ = np.linalg.qr(rng.standard_normal((n, k)))
    s_true = np.array([2.0 ** (-i * 0.5) for i in range(k)], dtype=np.float32) * 50.0
    a = (u_true * s_true) @ v_true.T

    # NumPy version
    u_rand, s_rand, vh_rand = randomized_svd(a, rank=k, n_iter=2, random_state=42)
    assert u_rand.shape == (m, k)
    assert s_rand.shape == (k,)
    assert vh_rand.shape == (k, n)

    # Check singular values match true singular values closely
    np.testing.assert_allclose(s_rand, s_true, rtol=1e-3, atol=1e-3)

    # Check reconstruction error
    a_rec = (u_rand * s_rand) @ vh_rand
    rel_err = np.linalg.norm(a - a_rec) / np.linalg.norm(a)
    assert rel_err < 1e-4

    # Orthonormality check
    np.testing.assert_allclose(u_rand.T @ u_rand, np.eye(k), atol=1e-5)
    np.testing.assert_allclose(vh_rand @ vh_rand.T, np.eye(k), atol=1e-5)

    # Test PyTorch version
    a_torch = torch.from_numpy(a)
    u_pt, s_pt, vh_pt = randomized_svd(a_torch, rank=k, n_iter=2, random_state=42)
    np.testing.assert_allclose(s_pt.numpy(), s_true, rtol=1e-3, atol=1e-3)


def test_power_iterations_improve_convergence():
    """Verify that power iterations (q=2) yield strictly better approximation than q=0."""
    rng = np.random.default_rng(2026)
    m, n, k = 100, 80, 10
    # Spectrum with slow decay and noise
    s_decay = 1.0 / (1.0 + np.arange(50))
    u_base, _ = np.linalg.qr(rng.standard_normal((m, 50)))
    v_base, _ = np.linalg.qr(rng.standard_normal((n, 50)))
    a = (u_base * s_decay) @ v_base.T

    # SVD with 0 iterations
    u_0, s_0, vh_0 = randomized_svd(a, rank=k, n_iter=0, n_oversamples=2, random_state=1)
    err_0 = np.linalg.norm(a - (u_0 * s_0) @ vh_0)

    # SVD with 2 iterations
    u_2, s_2, vh_2 = randomized_svd(a, rank=k, n_iter=2, n_oversamples=2, random_state=1)
    err_2 = np.linalg.norm(a - (u_2 * s_2) @ vh_2)

    assert err_2 < err_0, f"Power iterations failed to improve error: err_2={err_2}, err_0={err_0}"


def test_streaming_end_to_end_with_model(tmp_path: Path):
    """Verify end-to-end streaming context compilation creates valid PEFT adapter."""
    model = tiny_model()
    tokenizer = ByteTokenizer()
    compiler = EngramCompiler(model, tokenizer)

    # Long text: 600 characters
    long_text = "The quick brown fox jumps over the lazy dog. " * 15
    out_dir = compiler.compile_stream(
        context_text=long_text,
        output_dir=str(tmp_path / "streaming_adapter"),
        chunk_size=64,
        rank=4,
    )

    adapter_path = Path(out_dir)
    assert (adapter_path / "adapter_model.safetensors").is_file()
    assert (adapter_path / "adapter_config.json").is_file()
    assert compiler.last_prompt_tokens is not None and compiler.last_prompt_tokens > 64

    # Test loading with inject_engram
    peft_model = EngramCompiler.inject_engram(model, str(adapter_path))
    assert peft_model is not None
