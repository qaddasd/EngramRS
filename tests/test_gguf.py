from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs import EngramCompiler
from engram_rs.gguf import (
    GGMLType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
    export_lora_to_gguf,
    peft_to_gguf_tensor_name,
)


class DummyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        token_ids = [1, *(byte % 200 + 2 for byte in text.encode("utf-8"))]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def create_tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        _name_or_path="tiny-llama-gguf-test",
    )
    return LlamaForCausalLM(config)


def test_gguf_writer_and_reader_roundtrip(tmp_path: Path):
    """Test pure-Python GGUFWriter serialization and GGUFReader verification."""
    writer = GGUFWriter(alignment=32)

    # Add metadata
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_uint32("general.alignment", 32)
    writer.add_float32("adapter.lora.alpha", 16.0)
    writer.add_bool("general.test_flag", True)
    writer.add_array("general.tags", GGUFValueType.STRING, ["memory", "fast", "lora"])

    # Add tensors
    arr1 = np.arange(12, dtype=np.float32).reshape(3, 4)
    arr2 = np.ones((5, 2), dtype=np.float32) * 3.14

    writer.add_tensor("blk.0.attn_output.weight.lora_a", arr1, raw_dtype=GGMLType.F32)
    writer.add_tensor("blk.0.attn_output.weight.lora_b", arr2, raw_dtype=GGMLType.F16)

    gguf_path = tmp_path / "test.gguf"
    writer.write_to_file(gguf_path)

    assert gguf_path.is_file()

    # Read back and verify
    reader = GGUFReader(gguf_path)
    assert reader.version == 3
    assert reader.tensor_count == 2
    assert reader.metadata["general.type"] == "adapter"
    assert reader.metadata["adapter.type"] == "lora"
    assert reader.metadata["adapter.lora.alpha"] == 16.0
    assert reader.metadata["general.test_flag"] is True
    assert reader.metadata["general.tags"] == ["memory", "fast", "lora"]

    # Verify tensor 1 (F32)
    t1_info = reader.tensors["blk.0.attn_output.weight.lora_a"]
    assert t1_info["dtype"] == GGMLType.F32
    # In GGML, dimensions are innermost-first (reversed from NumPy [3, 4] -> [4, 3])
    assert t1_info["shape"] == [4, 3]
    assert t1_info["offset"] % 32 == 0
    t1_loaded = reader.get_tensor_data("blk.0.attn_output.weight.lora_a")
    assert np.allclose(t1_loaded, arr1)

    # Verify tensor 2 (F16)
    t2_info = reader.tensors["blk.0.attn_output.weight.lora_b"]
    assert t2_info["dtype"] == GGMLType.F16
    assert t2_info["shape"] == [2, 5]
    assert t2_info["offset"] % 32 == 0
    t2_loaded = reader.get_tensor_data("blk.0.attn_output.weight.lora_b")
    assert np.allclose(t2_loaded, arr2, atol=1e-2)


def test_peft_to_gguf_tensor_name_mapping():
    # Attention layers
    assert (
        peft_to_gguf_tensor_name("base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight")
        == "blk.0.attn_output.weight.lora_a"
    )
    assert (
        peft_to_gguf_tensor_name("base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight")
        == "blk.0.attn_output.weight.lora_b"
    )
    assert (
        peft_to_gguf_tensor_name("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight")
        == "blk.3.attn_q.weight.lora_a"
    )

    # MLP layers
    assert (
        peft_to_gguf_tensor_name("base_model.model.model.layers.5.mlp.down_proj.lora_B.weight")
        == "blk.5.ffn_down.weight.lora_b"
    )

    # Raw names
    assert (
        peft_to_gguf_tensor_name(
            "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight", standard=False
        )
        == "base_model.model.model.layers.0.self_attn.o_proj.lora_a"
    )


def test_export_lora_to_gguf_e2e(tmp_path: Path):
    """End-to-end test: compile an adapter with EngramCompiler and export to GGUF LoRA."""
    model = create_tiny_model()
    compiler = EngramCompiler(model, DummyTokenizer())

    adapter_dir = compiler.compile_text(
        "Cosmodrome Baikonur site launch authorization key 8841-GAMMA.",
        str(tmp_path / "adapter"),
        rank=2,
    )

    gguf_output = tmp_path / "adapter.gguf"
    result = export_lora_to_gguf(
        adapter_dir=adapter_dir,
        output_file=gguf_output,
        out_type="f32",
        architecture="llama",
    )

    assert Path(result) == gguf_output.resolve()
    assert gguf_output.is_file()

    reader = GGUFReader(gguf_output)
    assert reader.metadata["general.type"] == "adapter"
    assert reader.metadata["adapter.type"] == "lora"
    assert reader.metadata["general.architecture"] == "llama"
    assert reader.metadata["adapter.lora.alpha"] == 2.0
    assert reader.metadata["general.base_model.0.name"] == "tiny-llama-gguf-test"

    # Verify tensor names and shapes
    assert "blk.0.attn_output.weight.lora_a" in reader.tensors
    assert "blk.0.attn_output.weight.lora_b" in reader.tensors
    assert "blk.1.attn_output.weight.lora_a" in reader.tensors
    assert "blk.1.attn_output.weight.lora_b" in reader.tensors

    # Verify numerical match with safetensors
    from safetensors.torch import load_file

    st = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    orig_a = st["base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"].cpu().numpy()
    gguf_a = reader.get_tensor_data("blk.0.attn_output.weight.lora_a")
    assert np.allclose(orig_a, gguf_a, atol=1e-6)
