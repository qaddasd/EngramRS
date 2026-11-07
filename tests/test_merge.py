from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import EngramCompiler
from engram_rs.merge import (
    compute_delta_w,
    extract_lora_pairs,
    find_target_parameter,
    merge_adapter_to_base,
    merge_lora_into_model,
)


class DummyByteTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        token_ids = [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "Decoded response"


def create_tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        _name_or_path="tiny-llama-merge-test",
    )
    return LlamaForCausalLM(config)


def test_compute_delta_w_numerical_accuracy():
    r = 4
    alpha = 8.0  # scaling = 2.0
    in_features = 6
    out_features = 10

    rng = torch.Generator().manual_seed(42)
    lora_a = torch.randn(r, in_features, generator=rng)
    lora_b = torch.randn(out_features, r, generator=rng)

    delta_w = compute_delta_w(lora_a, lora_b, alpha=alpha, r=r)

    assert delta_w.shape == (out_features, in_features)
    expected = (lora_b @ lora_a) * 2.0
    assert torch.allclose(delta_w, expected, atol=1e-6)

    # Test fan_in_fan_out
    delta_w_transposed = compute_delta_w(lora_a, lora_b, alpha=alpha, r=r, fan_in_fan_out=True)
    assert delta_w_transposed.shape == (in_features, out_features)
    assert torch.allclose(delta_w_transposed, expected.t(), atol=1e-6)


def test_compute_delta_w_validation():
    a = torch.randn(2, 4)
    b = torch.randn(4, 2)
    with pytest.raises(ValueError, match="rank r must be positive"):
        compute_delta_w(a, b, alpha=4.0, r=0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        compute_delta_w(a, b, alpha=-1.0, r=2)


def test_zero_grad_merge_matches_peft_logits(tmp_path: Path):
    """Verify that merged base weights produce logits identical to PEFT injected adapter."""
    torch.manual_seed(42)
    base_model = create_tiny_llama()
    base_model.eval()

    torch.manual_seed(42)
    model_to_merge = create_tiny_llama()
    model_to_merge.eval()

    # Compile an actual Engram adapter
    compiler = EngramCompiler(base_model, DummyByteTokenizer())
    adapter_dir = compiler.compile_text(
        "Critical nuclear physics formula E=mc^2 observed in Kazakh steppe.",
        str(tmp_path / "adapter"),
        rank=2,
    )

    # 1. Obtain PEFT forward outputs
    peft_model = EngramCompiler.inject_engram(base_model, adapter_dir)
    peft_model.eval()
    tokenizer = DummyByteTokenizer()
    inputs = tokenizer("Prompt query", return_tensors="pt")

    with torch.no_grad():
        peft_outputs = peft_model(**inputs, use_cache=False)

    # 2. Perform analytical zero-grad merge on a separate fresh copy of the base model
    merged_model = merge_lora_into_model(model_to_merge, adapter_dir)
    assert merged_model is model_to_merge

    with torch.no_grad():
        merged_outputs = merged_model(**inputs, use_cache=False)

    # Logits should match to high precision
    assert torch.allclose(peft_outputs.logits, merged_outputs.logits, atol=1e-5), (
        f"Max difference: {torch.max(torch.abs(peft_outputs.logits - merged_outputs.logits)).item()}"
    )


def test_merge_adapter_to_base_saves_standalone_hf_model(tmp_path: Path):
    """Verify saving a standalone model that can be loaded directly without PEFT."""
    base_model = create_tiny_llama()
    compiler = EngramCompiler(base_model, DummyByteTokenizer())
    adapter_dir = compiler.compile_text(
        "Satellite telemetry station Sector 7-G.",
        str(tmp_path / "adapter"),
        rank=2,
    )

    output_dir = tmp_path / "merged_model"
    result_path = merge_adapter_to_base(
        base_model_name_or_path=base_model,
        adapter_path=adapter_dir,
        output_dir=output_dir,
    )

    assert Path(result_path) == output_dir.resolve()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "model.safetensors").is_file()

    # Load with standard AutoModelForCausalLM
    loaded = AutoModelForCausalLM.from_pretrained(str(output_dir))
    assert isinstance(loaded, LlamaForCausalLM)
    # Check that model forward works directly
    inputs = DummyByteTokenizer()("Test verification prompt")
    with torch.no_grad():
        out = loaded(**inputs)
    assert out.logits.shape[:2] == inputs["input_ids"].shape


def test_merge_rejects_missing_files(tmp_path: Path):
    model = create_tiny_llama()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing adapter configuration"):
        merge_lora_into_model(model, empty_dir)


def test_combine_lora_adapters_heterogeneous_scaling(tmp_path: Path):
    from safetensors.torch import save_file, load_file
    from engram_rs.merge import combine_lora_adapters

    dir_1 = tmp_path / "ad1"
    dir_2 = tmp_path / "ad2"
    dir_1.mkdir()
    dir_2.mkdir()

    d_in = 8
    d_out = 8
    r1, a1 = 2, 8.0   # scale = 4.0
    r2, a2 = 4, 2.0   # scale = 0.5

    rng = torch.Generator().manual_seed(123)
    a1_mat = torch.randn(r1, d_in, generator=rng)
    b1_mat = torch.randn(d_out, r1, generator=rng)

    a2_mat = torch.randn(r2, d_in, generator=rng)
    b2_mat = torch.randn(d_out, r2, generator=rng)

    key_a = "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"
    key_b = "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"

    save_file({key_a: a1_mat, key_b: b1_mat}, str(dir_1 / "adapter_model.safetensors"))
    with open(dir_1 / "adapter_config.json", "w") as f:
        json.dump({"r": r1, "lora_alpha": a1}, f)

    save_file({key_a: a2_mat, key_b: b2_mat}, str(dir_2 / "adapter_model.safetensors"))
    with open(dir_2 / "adapter_config.json", "w") as f:
        json.dump({"r": r2, "lora_alpha": a2}, f)

    # Compute expected Delta W sum: (a1/r1)*(B1 @ A1) + (a2/r2)*(B2 @ A2)
    expected_delta_w = (a1 / r1) * (b1_mat @ a1_mat) + (a2 / r2) * (b2_mat @ a2_mat)

    # Merge into single adapter with target_rank=6, target_alpha=12.0 (scale = 2.0)
    merged_dir = tmp_path / "merged"
    combine_lora_adapters([dir_1, dir_2], merged_dir, target_rank=6, target_alpha=12.0)

    merged_tensors = load_file(str(merged_dir / "adapter_model.safetensors"))
    with open(merged_dir / "adapter_config.json") as f:
        m_cfg = json.load(f)

    m_scale = m_cfg["lora_alpha"] / m_cfg["r"]
    actual_delta_w = m_scale * (merged_tensors[key_b] @ merged_tensors[key_a])

    assert torch.allclose(actual_delta_w, expected_delta_w, atol=1e-4)
