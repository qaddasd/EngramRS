from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from engram_rs.compiler import EngramCompiler
from engram_rs.merge import combine_lora_adapters, unlearn_concept_from_adapter
from engram_rs.profiler import LayerSensitivityProfiler
from engram_rs.streaming import StreamingCovarianceAccumulator


class DummyByteTokenizer:
    """Deterministic byte tokenizer for unit testing without HF Hub network access."""

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        encoded = [b % 64 for b in text.encode("utf-8")]
        if not encoded:
            encoded = [0]
        tensor = torch.tensor([encoded], dtype=torch.long)
        return {"input_ids": tensor}


class MockDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 16, intermediate_size: int = 32) -> None:
        super().__init__()
        self.input_layernorm = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.d_model = d_model
        self.intermediate_size = intermediate_size

        with torch.no_grad():
            self.input_layernorm.weight.copy_(torch.eye(d_model))
            self.self_attn.o_proj.weight.copy_(torch.eye(d_model) * 0.5)
            self.mlp.down_proj.weight.copy_(torch.ones(d_model, intermediate_size) * 0.1)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn.o_proj(normed)
        # Mock MLP activation
        bsz, seq_len, _ = hidden_states.shape
        m_act = torch.ones(bsz, seq_len, self.intermediate_size, device=hidden_states.device)
        mlp_out = self.mlp.down_proj(m_act)
        return hidden_states + attn_out + mlp_out


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 4, d_model: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MockDecoderLayer(d_model) for _ in range(num_layers)])
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, **kwargs):
        bsz, seq_len = input_ids.shape
        h = torch.ones(bsz, seq_len, self.d_model, dtype=torch.float32)
        for layer in self.layers:
            h = layer(h)
        return type("Output", (), {"logits": torch.randn(bsz, seq_len, 64), "last_hidden_state": h})()


def test_contrastive_delta_and_surprise_weighting(tmp_path: Path):
    model = MockModel(num_layers=4, d_model=16)
    tokenizer = DummyByteTokenizer()
    compiler = EngramCompiler(model, tokenizer)

    out_dir = tmp_path / "contrastive_adapter"
    res = compiler.compile_text(
        context_text="The confidential access token is ALPHA-99.",
        output_dir=str(out_dir),
        rank=4,
        contrastive_delta=True,
        surprise_weighting=True,
        scale_depth=True,
    )
    assert Path(res).is_dir()
    assert (out_dir / "adapter_model.safetensors").is_file()
    assert (out_dir / "adapter_config.json").is_file()

    with open(out_dir / "adapter_config.json", "r") as f:
        cfg = json.load(f)
    assert cfg["r"] == 4
    assert set(cfg["target_modules"]) == {"o_proj", "down_proj"}


def test_streaming_overlap_tokens(tmp_path: Path):
    model = MockModel(num_layers=2, d_model=16)
    tokenizer = DummyByteTokenizer()
    compiler = EngramCompiler(model, tokenizer)

    out_dir = tmp_path / "streaming_adapter"
    res = compiler.compile_stream(
        context_text="A" * 500,
        output_dir=str(out_dir),
        chunk_size=100,
        overlap_tokens=20,
        rank=2,
    )
    assert Path(res).is_dir()
    assert (out_dir / "adapter_model.safetensors").is_file()


def test_combine_lora_adapters(tmp_path: Path):
    model = MockModel(num_layers=2, d_model=16)
    tokenizer = DummyByteTokenizer()
    compiler = EngramCompiler(model, tokenizer)

    dir_a = tmp_path / "adapter_a"
    dir_b = tmp_path / "adapter_b"
    compiler.compile_text("First document about physics.", str(dir_a), rank=4)
    compiler.compile_text("Second document about chemistry.", str(dir_b), rank=4)

    merged_dir = tmp_path / "merged_adapter"
    out = combine_lora_adapters([dir_a, dir_b], merged_dir, target_rank=4)
    assert Path(out).is_dir()
    assert (merged_dir / "adapter_model.safetensors").is_file()

    with open(merged_dir / "adapter_config.json", "r") as f:
        cfg = json.load(f)
    assert cfg["r"] == 4


def test_unlearn_concept(tmp_path: Path):
    from safetensors.torch import load_file

    model = MockModel(num_layers=2, d_model=16)
    tokenizer = DummyByteTokenizer()
    compiler = EngramCompiler(model, tokenizer)

    adapter_dir = tmp_path / "orig_adapter"
    compiler.compile_text("Confidential medical record 12345.", str(adapter_dir), rank=2)

    # Concept activations to scrub (e.g. 4 rows of 16 features)
    rng = np.random.default_rng(42)
    concept = rng.standard_normal((4, 16), dtype=np.float32)

    scrubbed_dir = tmp_path / "scrubbed_adapter"
    out = unlearn_concept_from_adapter(adapter_dir, concept, scrubbed_dir)
    assert Path(out).is_dir()

    tensors = load_file(str(scrubbed_dir / "adapter_model.safetensors"))
    for k, v in tensors.items():
        if ".lora_A.weight" in k and v.shape[1] == 16:
            # Concept response must be zero
            concept_t = torch.from_numpy(concept)
            response = concept_t @ v.t()
            max_val = float(response.abs().max())
            assert max_val < 1e-5


def test_layer_sensitivity_profiler():
    model = MockModel(num_layers=4, d_model=16)
    tokenizer = DummyByteTokenizer()

    scores = LayerSensitivityProfiler.profile(model, tokenizer, "Test prompt for sensitivity profiling.")
    assert len(scores) == 4
    for idx, s in scores.items():
        assert s >= 0.0

    top_2 = LayerSensitivityProfiler.top_k_layers(scores, top_k=2)
    assert len(top_2) == 2

    pattern = LayerSensitivityProfiler.adaptive_rank_pattern(scores, base_rank=16, min_rank=4, max_rank=32)
    assert len(pattern) == 8  # 4 layers * 2 targets
