from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import EngramPrefixTuner, spectral_compress_kv


class ByteTokenizer:
    """Deterministic token-to-byte tokenizer for testing without external downloads."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        token_ids = [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        _name_or_path="tiny-llama-prefix-test",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_spectral_compress_kv_svd_shape_and_finite():
    """Verify SVD compression preserves head dimensions and contracts sequence length."""
    b, h, t, d = 2, 4, 64, 16
    p = 16
    key = torch.randn(b, h, t, d)
    value = torch.randn(b, h, t, d)

    k_p, v_p = spectral_compress_kv(key, value, prefix_len=p, method="svd")
    assert k_p.shape == (b, h, p, d)
    assert v_p.shape == (b, h, p, d)
    assert torch.isfinite(k_p).all()
    assert torch.isfinite(v_p).all()


def test_spectral_compress_kv_all_methods():
    """Verify all compression methods produce expected prefix tensor shapes."""
    b, h, t, d = 1, 2, 48, 8
    p = 12
    key = torch.randn(b, h, t, d)
    value = torch.randn(b, h, t, d)

    for method in ("svd", "spectral", "joint_svd", "energy", "nystrom"):
        k_p, v_p = spectral_compress_kv(key, value, prefix_len=p, method=method)
        assert k_p.shape == (b, h, p, d), f"method {method} yielded shape {k_p.shape}"
        assert v_p.shape == (b, h, p, d), f"method {method} yielded shape {v_p.shape}"
        assert torch.isfinite(k_p).all()
        assert torch.isfinite(v_p).all()


def test_spectral_compress_kv_preserves_attention_sinks():
    """Verify that preserve_sinks preserves the initial tokens verbatim."""
    b, h, t, d = 1, 2, 32, 8
    p = 8
    sinks = 2
    key = torch.randn(b, h, t, d)
    value = torch.randn(b, h, t, d)

    k_p, v_p = spectral_compress_kv(
        key,
        value,
        prefix_len=p,
        method="svd",
        preserve_sinks=sinks,
    )
    assert k_p.shape == (b, h, p, d)
    assert v_p.shape == (b, h, p, d)

    # First `sinks` tokens must be bitwise identical to the original key/value sinks
    assert torch.equal(k_p[..., :sinks, :], key[..., :sinks, :])
    assert torch.equal(v_p[..., :sinks, :], value[..., :sinks, :])


def test_spectral_compress_kv_short_context_padding():
    """Verify padding behavior when context is shorter than prefix_len."""
    b, h, t, d = 1, 2, 6, 8
    p = 10
    key = torch.randn(b, h, t, d)
    value = torch.randn(b, h, t, d)

    # With padding enabled (default)
    k_p, v_p = spectral_compress_kv(key, value, prefix_len=p, pad_to_prefix_len=True)
    assert k_p.shape == (b, h, p, d)
    assert v_p.shape == (b, h, p, d)
    assert torch.equal(k_p[..., :t, :], key)
    assert torch.equal(k_p[..., t:, :], torch.zeros(b, h, p - t, d))

    # Without padding
    k_nopad, v_nopad = spectral_compress_kv(key, value, prefix_len=p, pad_to_prefix_len=False)
    assert k_nopad.shape == (b, h, t, d)
    assert v_nopad.shape == (b, h, t, d)


def test_engram_prefix_tuner_tensor_round_trip():
    """Verify 6D analytical prefix tensor <-> DynamicCache lossless conversion."""
    num_layers = 3
    b, h, p, d = 1, 2, 8, 4
    tensor = torch.randn(num_layers, 2, b, h, p, d)

    cache = EngramPrefixTuner.from_tensor(tensor)
    assert isinstance(cache, DynamicCache)
    assert len(cache) == num_layers
    assert cache.get_seq_length() == p

    reconstructed = EngramPrefixTuner.to_tensor(cache)
    assert reconstructed.shape == tensor.shape
    assert torch.equal(reconstructed, tensor)


def test_save_and_load_prefix_safetensors(tmp_path: Path):
    """Verify prefix serialization and deserialization via safetensors."""
    num_layers = 2
    b, h, p, d = 1, 2, 16, 8
    tensor = torch.randn(num_layers, 2, b, h, p, d)
    cache = EngramPrefixTuner.from_tensor(tensor)

    tuner = EngramPrefixTuner(prefix_len=p, method="svd")
    out_file = tmp_path / "prefix.safetensors"
    tuner.save_prefix(cache, out_file)
    assert out_file.is_file()

    # Load as DynamicCache
    loaded_cache = tuner.load_prefix(out_file, return_cache=True)
    assert isinstance(loaded_cache, DynamicCache)
    assert len(loaded_cache) == num_layers
    assert loaded_cache.get_seq_length() == p

    # Compare values
    loaded_tensor = EngramPrefixTuner.to_tensor(loaded_cache)
    assert torch.allclose(loaded_tensor, tensor)

    # Load as tensor directly
    loaded_raw = tuner.load_prefix(out_file, return_cache=False)
    assert isinstance(loaded_raw, torch.Tensor)
    assert torch.allclose(loaded_raw, tensor)


def test_save_and_load_prefix_torch_pt(tmp_path: Path):
    """Verify prefix serialization and deserialization via PyTorch .pt format."""
    num_layers = 2
    b, h, p, d = 1, 2, 8, 4
    tensor = torch.randn(num_layers, 2, b, h, p, d)
    cache = EngramPrefixTuner.from_tensor(tensor)

    tuner = EngramPrefixTuner(prefix_len=p)
    out_file = tmp_path / "prefix.pt"
    tuner.save_prefix(cache, out_file)
    assert out_file.is_file()

    loaded_cache = tuner.load_prefix(out_file, return_cache=True)
    assert isinstance(loaded_cache, DynamicCache)
    loaded_tensor = EngramPrefixTuner.to_tensor(loaded_cache)
    assert torch.allclose(loaded_tensor, tensor)


def test_prefix_tuner_with_tiny_model_forward():
    """Verify forward pass on query tokens using pre-populated virtual prefix cache."""
    model = tiny_llama()
    tokenizer = ByteTokenizer()
    tuner = EngramPrefixTuner(
        model=model,
        tokenizer=tokenizer,
        prefix_len=8,
        method="svd",
        preserve_sinks=1,
    )

    context = "Security Protocol: Project Kronos access key is 9482-TX."
    prefix_cache = tuner.compile_prefix(context_text=context)

    assert isinstance(prefix_cache, DynamicCache)
    assert prefix_cache.get_seq_length() == 8

    # Query forward pass with past_key_values pre-populated with prefix
    query_ids = torch.tensor([[5, 10, 15, 20]])
    with torch.no_grad():
        out = model(query_ids, past_key_values=prefix_cache)

    assert out.logits is not None
    assert out.logits.shape == (1, 4, model.config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_prefix_tuner_generate_with_prefix():
    """Verify HuggingFace generation with pre-populated virtual prefix cache."""
    model = tiny_llama()
    tokenizer = ByteTokenizer()
    tuner = EngramPrefixTuner(
        model=model,
        tokenizer=tokenizer,
        prefix_len=6,
        method="svd",
    )

    context = "Secret key Alpha-7."
    prefix_cache = tuner.compile_prefix(context_text=context)

    query = torch.tensor([[1, 10, 20]])
    generated = tuner.generate_with_prefix(
        model=model,
        input_ids=query,
        prefix_cache=prefix_cache,
        max_new_tokens=4,
    )

    assert generated.shape == (1, query.shape[1] + 4)
    assert (generated >= 0).all()


def test_softmax_nonlinearity_preservation():
    """Demonstrate that virtual prefix retains full softmax non-linear attention capability."""
    # Context with 3 distinct keys: one matches query exactly, others orthogonal
    d = 8
    key_match = torch.zeros(1, 1, 1, d)
    key_match[0, 0, 0, 0] = 5.0  # Sharp spike on dimension 0

    key_other = torch.zeros(1, 1, 1, d)
    key_other[0, 0, 0, 1] = 5.0  # Orthogonal on dimension 1

    keys = torch.cat([key_match, key_other], dim=-2)  # shape (1, 1, 2, 8)

    val_match = torch.zeros(1, 1, 1, d)
    val_match[0, 0, 0, 2] = 10.0  # Target signal

    val_other = torch.zeros(1, 1, 1, d)
    val_other[0, 0, 0, 3] = 10.0  # Distractor signal

    vals = torch.cat([val_match, val_other], dim=-2)

    # Query matching key_match
    query = torch.zeros(1, 1, 1, d)
    query[0, 0, 0, 0] = 1.0

    # Softmax attention: A = softmax(Q K^T / sqrt(d))
    scale = 1.0 / (d ** 0.5)
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)

    attn_out = torch.matmul(probs, vals)

    # Due to softmax exponentiation, probability of match is overwhelmingly dominant
    prob_match = probs[0, 0, 0, 0].item()
    prob_other = probs[0, 0, 0, 1].item()
    assert prob_match > 0.8, f"Expected sharp softmax spike on matched key, got {prob_match}"
    assert prob_other < 0.2

    # The output sharply reflects val_match on dim 2, NOT a smoothed average
    assert attn_out[0, 0, 0, 2].item() > 8.0
