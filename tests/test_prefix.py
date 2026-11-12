from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import EngramPrefixTuner, spectral_compress_kv
from engram_rs.prefix import (
    _HAS_RUST_ROPE,
    _rope_rotate_torch,
    rope_apply_kv,
    rope_inverse_kv,
    select_compress_kv,
    snapkv_importance,
)


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


# ---------------------------------------------------------------------------
# RoPE correctness (fix C6/M6): selection + re-anchoring must match HuggingFace.
# ---------------------------------------------------------------------------


def _hf_rotary(model):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    return model.model.rotary_emb, apply_rotary_pos_emb


def test_rope_apply_matches_huggingface():
    """Our forward RoPE must equal HF apply_rotary_pos_emb at arbitrary positions."""
    model = tiny_llama()
    rotary, hf_apply = _hf_rotary(model)
    base = float(model.config.rope_theta)
    b, h, t, d = 1, 2, 7, model.config.hidden_size // model.config.num_attention_heads
    k_pre = torch.randn(b, h, t, d)
    positions = torch.arange(t)

    cos, sin = rotary(k_pre, positions.unsqueeze(0))
    _, k_hf = hf_apply(k_pre, k_pre, cos, sin)
    k_mine = rope_apply_kv(k_pre, positions, base=base, partial_rotary_factor=1.0)

    torch.testing.assert_close(k_mine, k_hf, atol=1e-5, rtol=1e-5)


def test_rope_inverse_undoes_huggingface():
    """Our inverse RoPE must recover the pre-RoPE key from HF's post-RoPE key."""
    model = tiny_llama()
    rotary, hf_apply = _hf_rotary(model)
    base = float(model.config.rope_theta)
    b, h, t, d = 1, 2, 9, model.config.hidden_size // model.config.num_attention_heads
    k_pre = torch.randn(b, h, t, d)
    positions = torch.arange(t)

    cos, sin = rotary(k_pre, positions.unsqueeze(0))
    _, k_hf = hf_apply(k_pre, k_pre, cos, sin)
    k_back = rope_inverse_kv(k_hf, positions, base=base, partial_rotary_factor=1.0)

    torch.testing.assert_close(k_back, k_pre, atol=1e-5, rtol=1e-5)


def test_rope_round_trip_identity_at_shifted_positions():
    """Invert at p_orig then re-apply at p_new equals a direct HF rotation at p_new."""
    model = tiny_llama()
    rotary, hf_apply = _hf_rotary(model)
    base = float(model.config.rope_theta)
    b, h, t, d = 1, 2, 5, model.config.hidden_size // model.config.num_attention_heads
    k_pre = torch.randn(b, h, t, d)
    p_orig = torch.tensor([3, 11, 40, 7, 99])
    p_new = torch.arange(t)

    cos_o, sin_o = rotary(k_pre, p_orig.unsqueeze(0))
    _, k_orig = hf_apply(k_pre, k_pre, cos_o, sin_o)  # HF key at original positions

    k_pre_back = rope_inverse_kv(k_orig, p_orig, base=base)
    k_new = rope_apply_kv(k_pre_back, p_new, base=base)

    cos_n, sin_n = rotary(k_pre, p_new.unsqueeze(0))
    _, k_ref = hf_apply(k_pre, k_pre, cos_n, sin_n)  # HF key at new positions
    torch.testing.assert_close(k_new, k_ref, atol=1e-4, rtol=1e-4)


def test_select_compress_kv_keeps_exact_tokens_and_sinks():
    """Selection must keep exact original rows (no blending) and align K with V."""
    b, h, t, d = 1, 2, 40, 8
    p, sinks = 10, 2
    key = torch.randn(b, h, t, d)
    value = torch.randn(b, h, t, d)

    k_sel, v_sel, idx = select_compress_kv(
        key, value, prefix_len=p, preserve_sinks=sinks, return_indices=True
    )
    assert k_sel.shape == (b, h, p, d)
    assert idx.shape == (p,)
    # Sinks preserved verbatim.
    assert torch.equal(k_sel[..., :sinks, :], key[..., :sinks, :])
    assert torch.equal(v_sel[..., :sinks, :], value[..., :sinks, :])
    # Every selected row is an exact original row, K and V at the same index.
    for i in range(p):
        j = int(idx[i])
        assert torch.equal(k_sel[..., i, :], key[..., j, :])
        assert torch.equal(v_sel[..., i, :], value[..., j, :])


def test_full_retention_prefix_matches_plain_forward():
    """A prefix keeping all tokens must reproduce a plain full-context forward (C6 plumbing)."""
    model = tiny_llama()
    base = float(model.config.rope_theta)
    context = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4]])
    query = torch.tensor([[6, 10, 14]])
    n = context.shape[1]

    with torch.no_grad():
        plain = model(torch.cat([context, query], dim=1)).logits[:, n:, :]

    tuner = EngramPrefixTuner(model=model, prefix_len=n, preserve_sinks=1)
    cache = tuner.capture_kv_cache(input_ids=context)
    prefix = tuner.compress_kv_cache(cache, prefix_len=n, rope_config=(base, 1.0))
    inputs = tuner.prepare_generate_inputs(query, prefix)
    with torch.no_grad():
        prefixed = model(**inputs).logits

    torch.testing.assert_close(prefixed, plain, atol=1e-4, rtol=1e-4)


def test_compressed_prefix_keys_match_hf_repositioning():
    """The definitive C6 regression: compressed keys equal HF RoPE re-anchored to 0..B-1.

    Builds an independent reference using HF's rotary embedding (inverse via -sin,
    forward via +sin) and compares it to the tuner's compiled prefix. A RoPE
    convention mismatch or phase collapse would make these diverge.
    """
    model = tiny_llama()
    rotary, hf_apply = _hf_rotary(model)
    base = float(model.config.rope_theta)
    context = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 12, 6]])
    n = context.shape[1]
    p = 5

    tuner = EngramPrefixTuner(model=model, prefix_len=p, preserve_sinks=1)
    cache = tuner.capture_kv_cache(input_ids=context)
    compiled = tuner.compress_kv_cache(cache, prefix_len=p, rope_config=(base, 1.0))

    k_full, _ = cache[0]  # [B, H, N, D] post-RoPE at original positions
    b, h, _, d = k_full.shape
    importance = (k_full * k_full).sum(-1).mean(dim=(0, 1))
    sinks = 1
    tail = importance[sinks:]
    _, top = torch.topk(tail, p - sinks)
    idx = torch.cat([torch.arange(sinks), torch.sort(top).values + sinks])
    k_sel_orig = k_full[:, :, idx, :]

    # Independent HF reference: invert at original positions, re-apply at 0..P-1.
    cos_o, sin_o = rotary(k_sel_orig, idx.unsqueeze(0))
    _, k_pre = hf_apply(k_sel_orig, k_sel_orig, cos_o, -sin_o)
    new_pos = torch.arange(p)
    cos_n, sin_n = rotary(k_pre, new_pos.unsqueeze(0))
    _, k_ref = hf_apply(k_pre, k_pre, cos_n, sin_n)

    k_compiled, _ = compiled[0]
    torch.testing.assert_close(k_compiled, k_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _HAS_RUST_ROPE, reason="native RoPE extension not built")
def test_rust_rope_matches_torch_reference():
    """The Rust core RoPE must agree with the torch reference (CPU float32 path)."""
    k = torch.randn(1, 2, 7, 8)
    pos = torch.arange(7)
    rust_apply = rope_apply_kv(k, pos, base=10000.0)  # routes through Rust on CPU f32
    ref_apply = _rope_rotate_torch(k, pos, 10000.0, 1.0, inverse=False)
    torch.testing.assert_close(rust_apply, ref_apply, atol=1e-5, rtol=1e-5)

    rust_inv = rope_inverse_kv(rust_apply, pos, base=10000.0)
    torch.testing.assert_close(rust_inv, k, atol=1e-5, rtol=1e-5)


def _tiny_llama_eager() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        _name_or_path="tiny-llama-snapkv-test",
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_snapkv_importance_shape_and_nonnegative():
    model = _tiny_llama_eager()
    input_ids = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 12, 6]])
    scores = snapkv_importance(model, input_ids, obs_window=4)
    assert scores.shape == (model.config.num_hidden_layers, input_ids.shape[1])
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_compress_kv_cache_explicit_scores_select_top_tokens():
    """With explicit scores, selection must keep the highest-scored exact tokens."""
    b, h, t, d = 1, 2, 10, 8
    key = torch.randn(b, h, t, d)
    value = torch.randn(b, h, t, d)
    cache = DynamicCache.from_legacy_cache(((key, value),))
    scores = torch.tensor([0.0, 1, 2, 100, 3, 4, 5, 200, 6, 7])

    tuner = EngramPrefixTuner(prefix_len=3, preserve_sinks=1)
    compiled = tuner.compress_kv_cache(cache, prefix_len=3, preserve_sinks=1, scores=scores)
    k_sel, v_sel = compiled[0]
    expected_idx = torch.tensor([0, 3, 7])  # sink 0 + top-2 tail (7=200, 3=100)
    torch.testing.assert_close(k_sel, key[:, :, expected_idx, :])
    torch.testing.assert_close(v_sel, value[:, :, expected_idx, :])


def test_compile_prefix_snapkv_produces_valid_cache():
    model = _tiny_llama_eager()
    input_ids = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 12, 6]])
    tuner = EngramPrefixTuner(model=model, prefix_len=5, preserve_sinks=1)
    prefix = tuner.compile_prefix(input_ids=input_ids, prefix_len=5, selection="snapkv")
    assert len(prefix) == model.config.num_hidden_layers
    for layer_idx in range(len(prefix)):
        k, v = prefix[layer_idx]
        assert k.shape[-2] == 5
        assert v.shape[-2] == 5
        assert torch.isfinite(k).all()
        assert torch.isfinite(v).all()


def test_snapkv_importance_rejects_non_eager():
    """SDPA/flash models return no attentions; SnapKV must raise, not silently fall back."""
    config = LlamaConfig(
        vocab_size=256, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 5, 9, 13, 2, 7]])
    with pytest.raises(RuntimeError):
        snapkv_importance(model, input_ids, obs_window=4)


def test_snapkv_importance_query_aware_scores_context_only():
    """Query-aware window scores ONLY context tokens -> shape (L, ctx_len)."""
    model = _tiny_llama_eager()
    ctx = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 12, 6]])
    query = torch.tensor([[4, 9, 2]])
    scores = snapkv_importance(model, ctx, obs_window=4, query_ids=query)
    assert scores.shape == (model.config.num_hidden_layers, ctx.shape[1])
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_snapkv_importance_query_aware_differs_from_agnostic():
    """Using the question as the observation window must change the scores."""
    model = _tiny_llama_eager()
    ctx = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 12, 6]])
    query = torch.tensor([[4, 9, 2]])
    agnostic = snapkv_importance(model, ctx, obs_window=4)
    aware = snapkv_importance(model, ctx, obs_window=4, query_ids=query)
    assert (agnostic - aware).abs().max() > 1e-6


def test_snapkv_importance_rejects_empty_query():
    model = _tiny_llama_eager()
    ctx = torch.tensor([[1, 5, 9, 13]])
    with pytest.raises(ValueError):
        snapkv_importance(model, ctx, query_ids=torch.empty((1, 0), dtype=torch.long))


def test_compile_prefix_query_ids_produces_valid_cache():
    model = _tiny_llama_eager()
    ctx = torch.tensor([[1, 5, 9, 13, 2, 7, 11, 4, 3, 8, 12, 6]])
    query = torch.tensor([[4, 9, 2]])
    tuner = EngramPrefixTuner(model=model, prefix_len=5, preserve_sinks=1)
    prefix = tuner.compile_prefix(
        input_ids=ctx, prefix_len=5, selection="snapkv", query_ids=query
    )
    assert len(prefix) == model.config.num_hidden_layers
    for layer_idx in range(len(prefix)):
        k, v = prefix[layer_idx]
        assert k.shape[-2] == 5
        assert v.shape[-2] == 5
        assert torch.isfinite(k).all()
        assert torch.isfinite(v).all()
