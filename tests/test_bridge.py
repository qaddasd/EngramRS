from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    from engram_rs import compile_all_layers_to_safetensors
except ImportError:
    compile_all_layers_to_safetensors = None

pytestmark = pytest.mark.skipif(
    compile_all_layers_to_safetensors is None,
    reason="the engram_rs native extension has not been built",
)


def read_safetensors(path: Path):
    contents = path.read_bytes()
    header_length = struct.unpack("<Q", contents[:8])[0]
    header = json.loads(contents[8 : 8 + header_length])
    data = contents[8 + header_length :]
    tensors = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        assert metadata["dtype"] == "F32"
        start, end = metadata["data_offsets"]
        values = np.frombuffer(data[start:end], dtype="<f4").reshape(metadata["shape"])
        tensors[name] = values
    return header, tensors


def test_numpy_bridge_compiles_and_writes_real_safetensors(tmp_path: Path):
    rng = np.random.default_rng(7)
    hidden_0 = rng.normal(size=(12, 4)).astype(np.float32)
    projection = rng.normal(size=(4, 4)).astype(np.float32) / 4.0
    attention_0 = hidden_0 @ projection
    hidden_1 = rng.normal(size=(12, 4)).astype(np.float32)
    attention_1 = hidden_1 @ projection * 0.5
    # Null-state token counts are independent; only d_model must agree.
    null_0 = rng.normal(size=(5, 4)).astype(np.float32) * 0.01
    null_1 = rng.normal(size=(7, 4)).astype(np.float32) * 0.01

    elapsed_ms = compile_all_layers_to_safetensors(
        [hidden_0, hidden_1],
        [attention_0, attention_1],
        output_dir=tmp_path,
        null_states=[null_0, null_1],
        rank=2,
        max_anchors=16,
        lambda_reg=1e-3,
        lambda_null=0.5,
        base_model_name_or_path="tiny-local-test-model",
    )

    assert isinstance(elapsed_ms, float)
    assert elapsed_ms >= 0.0
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "tiny-local-test-model"
    assert config["r"] == 2
    assert config["lora_alpha"] == 2
    assert config["target_modules"] == ["o_proj"]

    header, tensors = read_safetensors(tmp_path / "adapter_model.safetensors")
    assert len(tensors) == 4
    assert len(header) >= 4
    for layer in range(2):
        prefix = f"base_model.model.model.layers.{layer}.self_attn.o_proj.lora_"
        a = tensors[prefix + "A.weight"]
        b = tensors[prefix + "B.weight"]
        assert a.shape == (2, 4)
        assert b.shape == (4, 2)
        assert np.isfinite(a).all()
        assert np.isfinite(b).all()


def test_bridge_rejects_noncontiguous_arrays_before_compilation(tmp_path: Path):
    base = np.arange(48, dtype=np.float32).reshape(12, 4)
    noncontiguous = base[::2]
    assert not noncontiguous.flags.c_contiguous
    attention = np.ones_like(noncontiguous)
    with pytest.raises(ValueError, match=r"hidden_states\[0\].*C-contiguous"):
        compile_all_layers_to_safetensors(
            [noncontiguous],
            [attention],
            output_dir=tmp_path,
            rank=2,
            max_anchors=16,
            base_model_name_or_path="tiny-local-test-model",
        )


def test_bridge_validates_list_lengths_and_matrix_shapes(tmp_path: Path):
    hidden = np.ones((6, 4), dtype=np.float32)
    attention = np.ones((6, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="same number of layers"):
        compile_all_layers_to_safetensors(
            [hidden, hidden],
            [attention],
            output_dir=tmp_path,
            rank=2,
            base_model_name_or_path="tiny-local-test-model",
        )

    with pytest.raises(ValueError, match="layer 0 shape mismatch"):
        compile_all_layers_to_safetensors(
            [hidden],
            [np.ones((6, 3), dtype=np.float32)],
            output_dir=tmp_path,
            rank=2,
            base_model_name_or_path="tiny-local-test-model",
        )

    with pytest.raises(ValueError, match="null_states must contain one matrix per layer"):
        compile_all_layers_to_safetensors(
            [hidden],
            [attention],
            output_dir=tmp_path,
            null_states=[],
            rank=2,
            base_model_name_or_path="tiny-local-test-model",
        )

    with pytest.raises(ValueError, match="null_states has d_model=3, expected 4"):
        compile_all_layers_to_safetensors(
            [hidden],
            [attention],
            output_dir=tmp_path,
            null_states=[np.ones((3, 3), dtype=np.float32)],
            rank=2,
            base_model_name_or_path="tiny-local-test-model",
        )


def test_bridge_multi_target_and_spectral_energy(tmp_path: Path):
    rng = np.random.default_rng(42)
    # Layer 0 and 1
    # o_proj: d_in=4, d_out=4
    o_in_0 = rng.normal(size=(10, 4)).astype(np.float32)
    o_out_0 = (o_in_0 @ rng.normal(size=(4, 4)).astype(np.float32)).astype(np.float32)
    o_in_1 = rng.normal(size=(10, 4)).astype(np.float32)
    o_out_1 = (o_in_1 @ rng.normal(size=(4, 4)).astype(np.float32)).astype(np.float32)

    # down_proj: intermediate_size=8, d_model=4
    mlp_in_0 = rng.normal(size=(10, 8)).astype(np.float32)
    mlp_out_0 = (mlp_in_0 @ rng.normal(size=(8, 4)).astype(np.float32)).astype(np.float32)
    mlp_in_1 = rng.normal(size=(10, 8)).astype(np.float32)
    mlp_out_1 = (mlp_in_1 @ rng.normal(size=(8, 4)).astype(np.float32)).astype(np.float32)

    target_inputs = {
        "o_proj": [o_in_0, o_in_1],
        "down_proj": [mlp_in_0, mlp_in_1],
    }
    target_outputs = {
        "o_proj": [o_out_0, o_out_1],
        "down_proj": [mlp_out_0, mlp_out_1],
    }

    elapsed_ms = compile_all_layers_to_safetensors(
        output_dir=tmp_path,
        rank=4,
        max_anchors=16,
        lambda_reg=1e-3,
        lambda_null=0.0,
        base_model_name_or_path="multi-target-test-model",
        targets=["o_proj", "down_proj"],
        target_inputs=target_inputs,
        target_outputs=target_outputs,
        energy_threshold=0.95,
        min_rank=1,
        max_rank=4,
    )

    assert isinstance(elapsed_ms, float)
    assert elapsed_ms >= 0.0

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["target_modules"] == ["o_proj", "down_proj"]
    assert "r" in config

    _, tensors = read_safetensors(tmp_path / "adapter_model.safetensors")
    # 2 layers * 2 modules * 2 matrices (A and B) = 8 tensors
    assert len(tensors) == 8
    for layer in range(2):
        for module in ("self_attn.o_proj", "mlp.down_proj"):
            prefix = f"base_model.model.model.layers.{layer}.{module}.lora_"
            assert prefix + "A.weight" in tensors
            assert prefix + "B.weight" in tensors
            assert np.isfinite(tensors[prefix + "A.weight"]).all()
            assert np.isfinite(tensors[prefix + "B.weight"]).all()


def test_simd_centering_bf16_and_f32():
    from engram_rs.engram_rs import center_and_unpack_bf16_py, center_f32_py
    import torch

    # 2D f32
    f32_data = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    centered = center_f32_py(f32_data)
    np.testing.assert_allclose(centered, [[0.0, 0.0], [2.0, 3.0]])

    # 2D bf16
    bf16_tensor = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.bfloat16)
    u16_data = bf16_tensor.view(torch.uint16).numpy()
    unpacked = center_and_unpack_bf16_py(u16_data)
    np.testing.assert_allclose(unpacked, [[0.0, 0.0], [2.0, 3.0]])

