from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import EngramCompiler


class ByteTokenizer:
    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        token_ids = [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class CountingLlama(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def tiny_model() -> CountingLlama:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        _name_or_path="tiny-llama-local",
    )
    return CountingLlama(config)


def test_compile_text_runs_once_and_loads_through_peft(tmp_path: Path):
    model = tiny_model()
    model.train()
    compiler = EngramCompiler(model, ByteTokenizer())

    adapter_dir = compiler.compile_text(
        "Aktau laboratory Omega-9 launch code 7391-KZR.",
        str(tmp_path / "adapter"),
        rank=2,
    )

    assert model.forward_calls == 1
    assert model.training
    assert compiler.last_prompt_tokens and compiler.last_prompt_tokens > 1
    assert compiler.last_compile_time_ms is not None
    assert compiler.last_compile_time_ms >= 0.0
    assert Path(adapter_dir) == (tmp_path / "adapter").resolve()
    adapter_config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    assert adapter_config["task_type"] == "CAUSAL_LM"
    assert adapter_config["r"] == 2
    assert adapter_config["base_model_name_or_path"] == "tiny-llama-local"

    adapted = EngramCompiler.inject_engram(model, adapter_dir)
    lora_a = [parameter for name, parameter in adapted.named_parameters() if "lora_A" in name]
    # 2 layers * 2 targets (o_proj and down_proj) = 4 lora_A parameters
    assert len(lora_a) == 4
    assert set(adapter_config["target_modules"]) == {"o_proj", "down_proj"}
    assert any(torch.count_nonzero(parameter).item() for parameter in lora_a)
    inputs = ByteTokenizer()("Question only", return_tensors="pt")
    with torch.no_grad():
        outputs = adapted(**inputs, use_cache=False)
    assert outputs.logits.shape[:2] == inputs["input_ids"].shape


def test_compile_text_with_custom_targets_and_spectral_energy(tmp_path: Path):
    # 1. Single target: o_proj only
    model_o = tiny_model()
    compiler_o = EngramCompiler(model_o, ByteTokenizer())
    dir_o = compiler_o.compile_text(
        "Aktau laboratory Omega-9 launch code 7391-KZR.",
        str(tmp_path / "adapter_o"),
        rank=2,
        targets=("o_proj",),
    )
    cfg_o = json.loads((Path(dir_o) / "adapter_config.json").read_text())
    assert cfg_o["target_modules"] == ["o_proj"]
    adapted_o = EngramCompiler.inject_engram(model_o, dir_o)
    lora_a_o = [p for n, p in adapted_o.named_parameters() if "lora_A" in n]
    assert len(lora_a_o) == 2

    # 2. Single target: down_proj only
    model_down = tiny_model()
    compiler_down = EngramCompiler(model_down, ByteTokenizer())
    dir_down = compiler_down.compile_text(
        "Aktau laboratory Omega-9 launch code 7391-KZR.",
        str(tmp_path / "adapter_down"),
        rank=2,
        targets=("down_proj",),
    )
    cfg_down = json.loads((Path(dir_down) / "adapter_config.json").read_text())
    assert cfg_down["target_modules"] == ["down_proj"]
    adapted_down = EngramCompiler.inject_engram(model_down, dir_down)
    lora_a_down = [p for n, p in adapted_down.named_parameters() if "lora_A" in n]
    assert len(lora_a_down) == 2

    # 3. Multi-target with spectral energy allocation
    model_energy = tiny_model()
    compiler_energy = EngramCompiler(model_energy, ByteTokenizer())
    dir_energy = compiler_energy.compile_text(
        "Aktau laboratory Omega-9 launch code 7391-KZR.",
        str(tmp_path / "adapter_energy"),
        rank=4,
        targets=("o_proj", "down_proj"),
        energy_threshold=0.95,
        min_rank=1,
        max_rank=4,
    )
    cfg_energy = json.loads((Path(dir_energy) / "adapter_config.json").read_text())
    assert set(cfg_energy["target_modules"]) == {"o_proj", "down_proj"}
    assert "rank_pattern" in cfg_energy
    adapted_energy = EngramCompiler.inject_engram(model_energy, dir_energy)
    lora_a_energy = [p for n, p in adapted_energy.named_parameters() if "lora_A" in n]
    assert len(lora_a_energy) == 4
    inputs = ByteTokenizer()("Query", return_tensors="pt")
    with torch.no_grad():
        out = adapted_energy(**inputs, use_cache=False)
    assert out.logits.shape[:2] == inputs["input_ids"].shape


def test_compile_text_rejects_invalid_inputs(tmp_path: Path):
    compiler = EngramCompiler(tiny_model(), ByteTokenizer())
    with pytest.raises(ValueError, match="context_text"):
        compiler.compile_text("", str(tmp_path))
    with pytest.raises(ValueError, match="rank"):
        compiler.compile_text("valid context", str(tmp_path), rank=0)
    with pytest.raises(ValueError, match="output_dir"):
        compiler.compile_text("valid context", "", rank=2)
