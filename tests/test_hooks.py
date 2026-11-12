from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from engram_rs.hooks import EngramContextCatcher


class TupleProjection(nn.Linear):
    def forward(self, inputs: torch.Tensor):
        return super().forward(inputs), {"present_key_values": None}


class TinyAttention(nn.Module):
    def __init__(self, d_model: int, output_scale: float) -> None:
        super().__init__()
        self.o_proj = TupleProjection(d_model, d_model, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(d_model) * output_scale)


class TinyDecoderLayer(nn.Module):
    def __init__(self, d_model: int, input_scale: float, output_scale: float) -> None:
        super().__init__()
        self.input_layernorm = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.input_layernorm.weight.copy_(torch.eye(d_model) * input_scale)
        self.self_attn = TinyAttention(d_model, output_scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        attention_output, _ = self.self_attn.o_proj(normalized)
        return attention_output


class TinyBackbone(nn.Module):
    def __init__(self, layers: list[TinyDecoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class TinyHFModel(nn.Module):
    def __init__(self, d_model: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
        )
        self.model = TinyBackbone(
            [
                TinyDecoderLayer(d_model, input_scale=1.0, output_scale=2.0),
                TinyDecoderLayer(d_model, input_scale=3.0, output_scale=1.0),
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def centered(values: torch.Tensor) -> np.ndarray:
    values = values.detach().to(dtype=torch.float32, device="cpu")
    if values.ndim == 2:
        values = values - values[0:1]
    else:
        values = values - values[:, 0:1]
    return values.reshape(-1, values.shape[-1]).numpy()


def test_capture_sink_centers_each_batch_and_preserves_requested_layer_order():
    model = TinyHFModel()
    tokens = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 2.0) / 7.0

    with EngramContextCatcher(model, [1, 0]) as catcher:
        assert catcher.layer_indices == (1, 0)
        with torch.no_grad():
            output = model(tokens)
        hidden, attention = catcher.get_activations()

    assert torch.equal(output, tokens * 6.0)
    np.testing.assert_allclose(hidden[0], centered(tokens * 6.0))
    np.testing.assert_allclose(attention[0], centered(tokens * 6.0))
    np.testing.assert_allclose(hidden[1], centered(tokens))
    np.testing.assert_allclose(attention[1], centered(tokens * 2.0))

    for matrix in [*hidden, *attention]:
        assert matrix.shape == (6, 4)
        assert matrix.dtype == np.float32
        assert matrix.flags.c_contiguous
        np.testing.assert_array_equal(matrix[0], np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(matrix[3], np.zeros(4, dtype=np.float32))

    # Context exit removes hooks but deliberately leaves arrays usable.
    for layer in model.model.layers:
        assert len(layer.input_layernorm._forward_hooks) == 0
        assert len(layer.self_attn.o_proj._forward_hooks) == 0
    np.testing.assert_allclose(catcher.hidden_states[0], hidden[0])


def test_numpy_storage_survives_model_output_and_python_gc():
    model = TinyHFModel()
    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    with EngramContextCatcher(model, [0]) as catcher:
        output = model(tokens)
        hidden = catcher.hidden_states[0]
        expected = hidden.copy()
    del output, tokens
    gc.collect()
    np.testing.assert_array_equal(hidden, expected)
    assert hidden.base is not None


def test_two_dimensional_capture_uses_first_token_as_single_batch_sink():
    model = TinyHFModel()
    tokens = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1.0
    with EngramContextCatcher(model, [0]) as catcher:
        model(tokens)
        hidden, attention = catcher.get_activations()
    np.testing.assert_allclose(hidden[0], centered(tokens))
    np.testing.assert_allclose(attention[0], centered(tokens * 2.0))


def test_gqa_metadata_does_not_repeat_or_rotate_o_proj_outputs():
    model = TinyHFModel()
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    with EngramContextCatcher(model, [0]) as catcher:
        model(tokens)
        attention = catcher.attention_outputs[0]
        metadata = catcher.capture_metadata

    assert attention.shape == (6, 4)
    assert metadata["num_attention_heads"] == 4
    assert metadata["num_key_value_heads"] == 2
    assert metadata["num_query_groups"] == 2
    assert metadata["head_dim"] == 2
    assert metadata["o_proj_output_repeated"] is False
    assert metadata["rope_inversion_applied"] is False
    assert metadata["rope_inversion_required"] is False
    assert metadata["sink_centered_per_batch"] is True


def test_close_and_clear_remove_hooks_and_release_captures():
    model = TinyHFModel()
    tokens = torch.zeros(1, 2, 4)
    catcher = EngramContextCatcher(model, [0])
    with catcher:
        assert len(model.model.layers[0].input_layernorm._forward_hooks) == 0
        assert len(model.model.layers[0].self_attn.o_proj._forward_hooks) == 1
        model(tokens)
        catcher.get_activations()
    catcher.close()
    assert len(model.model.layers[0].input_layernorm._forward_hooks) == 0
    assert len(model.model.layers[0].self_attn.o_proj._forward_hooks) == 0
    with pytest.raises(RuntimeError, match="no activations"):
        catcher.get_activations()

    with catcher:
        assert len(model.model.layers[0].input_layernorm._forward_hooks) == 0
        catcher.clear()
        assert len(model.model.layers[0].input_layernorm._forward_hooks) == 0
        assert len(model.model.layers[0].self_attn.o_proj._forward_hooks) == 0


def test_missing_modules_and_missing_captures_have_clear_errors():
    class MissingProjectionLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.self_attn = nn.Module()

    class MissingProjectionBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([MissingProjectionLayer()])

    class MissingProjectionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = MissingProjectionBackbone()

    with pytest.raises(AttributeError, match="self_attn.o_proj"):
        EngramContextCatcher(MissingProjectionModel(), [0])

    model = TinyHFModel()
    with EngramContextCatcher(model, [0]) as catcher:
        assert catcher.layer_indices == (0,)
    with pytest.raises(RuntimeError, match="no activations"):
        catcher.get_activations()

    with EngramContextCatcher(model, [0]) as catcher:
        model.model.layers[0].input_layernorm(torch.zeros(1, 2, 4))
    with pytest.raises(RuntimeError, match="no activations"):
        catcher.get_activations()


def test_repeated_forward_is_rejected_instead_of_mixing_passes():
    model = TinyHFModel()
    tokens = torch.zeros(1, 2, 4)
    with EngramContextCatcher(model, [0]):
        model(tokens)
        with pytest.raises(RuntimeError, match="exactly one pass"):
            model(tokens)
    for layer in model.model.layers:
        assert not layer.input_layernorm._forward_hooks
        assert not layer.self_attn.o_proj._forward_hooks


def test_invalid_layer_selection_is_rejected():
    model = TinyHFModel()
    with pytest.raises(IndexError, match="out of range"):
        EngramContextCatcher(model, [2])
    with pytest.raises(ValueError, match="at least one"):
        EngramContextCatcher(model, [])
    with pytest.raises(ValueError, match="duplicate"):
        EngramContextCatcher(model, [0, 0])


class TinyMLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.ones(intermediate_size, d_model) * 0.5)
            self.up_proj.weight.copy_(torch.ones(intermediate_size, d_model) * 0.5)
            self.down_proj.weight.copy_(torch.ones(d_model, intermediate_size) * 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(m)


class TinyDecoderLayerWithMLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int) -> None:
        super().__init__()
        self.input_layernorm = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = TinyAttention(d_model, output_scale=1.0)
        self.mlp = TinyMLP(d_model, intermediate_size)
        with torch.no_grad():
            self.input_layernorm.weight.copy_(torch.eye(d_model))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        attn_out, _ = self.self_attn.o_proj(normalized)
        hidden = hidden + attn_out
        mlp_out = self.mlp(hidden)
        return hidden + mlp_out


class TinyModelWithMLP(nn.Module):
    def __init__(self, d_model: int = 4, intermediate_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=d_model,
        )
        self.model = TinyBackbone(
            [
                TinyDecoderLayerWithMLP(d_model, intermediate_size),
                TinyDecoderLayerWithMLP(d_model, intermediate_size),
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def test_mlp_and_multi_target_capture():
    model = TinyModelWithMLP(d_model=4, intermediate_size=8)
    tokens = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1.0) / 5.0

    with EngramContextCatcher(model) as catcher:
        assert catcher.targets == ("o_proj", "down_proj")
        model(tokens)
        all_acts = catcher.get_all_activations()

    assert set(all_acts.keys()) == {"o_proj", "down_proj"}
    hidden_states, attention_outputs = all_acts["o_proj"]
    mlp_inputs, mlp_outputs = all_acts["down_proj"]

    assert len(hidden_states) == 2
    assert len(mlp_inputs) == 2

    # o_proj: (batch*seq, d_model) -> (6, 4)
    assert hidden_states[0].shape == (6, 4)
    assert attention_outputs[0].shape == (6, 4)

    # down_proj: intermediate activation m(x) has intermediate_size=8, output has d_model=4
    assert mlp_inputs[0].shape == (6, 8)
    assert mlp_outputs[0].shape == (6, 4)

    # Sink-centered per batch: first token of batch 0 (index 0) and batch 1 (index 3) must be 0
    np.testing.assert_array_equal(mlp_inputs[0][0], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(mlp_inputs[0][3], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(mlp_outputs[0][0], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(mlp_outputs[0][3], np.zeros(4, dtype=np.float32))

    # Properties match
    np.testing.assert_array_equal(catcher.mlp_inputs[0], mlp_inputs[0])
    np.testing.assert_array_equal(catcher.mlp_outputs[0], mlp_outputs[0])


def test_target_selection_down_proj_only():
    model = TinyModelWithMLP(d_model=4, intermediate_size=8)
    tokens = torch.randn(1, 3, 4)

    with EngramContextCatcher(model, targets=("down_proj",)) as catcher:
        assert catcher.targets == ("down_proj",)
        layer = model.model.layers[0]
        assert len(layer.mlp.down_proj._forward_hooks) == 1
        assert len(layer.input_layernorm._forward_hooks) == 0
        assert len(layer.self_attn.o_proj._forward_hooks) == 0

        model(tokens)
        inputs, outputs = catcher.get_activations()
        assert inputs[0].shape == (3, 8)
        assert outputs[0].shape == (3, 4)


def test_target_selection_o_proj_only():
    model = TinyModelWithMLP(d_model=4, intermediate_size=8)
    tokens = torch.randn(1, 3, 4)

    with EngramContextCatcher(model, targets=("o_proj",)) as catcher:
        assert catcher.targets == ("o_proj",)
        layer = model.model.layers[0]
        assert len(layer.mlp.down_proj._forward_hooks) == 0
        assert len(layer.input_layernorm._forward_hooks) == 0
        assert len(layer.self_attn.o_proj._forward_hooks) == 1

        model(tokens)
        inputs, outputs = catcher.get_activations()
        assert inputs[0].shape == (3, 4)
        assert outputs[0].shape == (3, 4)


def test_invalid_targets_and_missing_mlp_error():
    model = TinyHFModel()  # has no mlp

    with pytest.raises(ValueError, match="unsupported target"):
        EngramContextCatcher(model, targets=("unknown_module",))

    with pytest.raises(ValueError, match="at least one"):
        EngramContextCatcher(model, targets=())

    with pytest.raises(AttributeError, match="mlp.down_proj"):
        EngramContextCatcher(model, targets=("down_proj",))


class DistinctBasisLayer(nn.Module):
    """o_proj input differs from input_layernorm output (as in real attention)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.input_layernorm = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = SimpleNamespace(
            o_proj=TupleProjection(d_model, d_model, bias=False)
        )
        with torch.no_grad():
            self.input_layernorm.weight.copy_(torch.eye(d_model) * 2.0)
            self.self_attn.o_proj.weight.copy_(torch.eye(d_model))
        self.last_o_proj_input: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        # Simulate attention mixing: o_proj sees a different basis than layernorm.
        attn_concat = normalized * 5.0 + 1.0
        self.last_o_proj_input = attn_concat.detach()
        out, _ = self.self_attn.o_proj(attn_concat)
        return out


class DistinctBasisModel(nn.Module):
    def __init__(self, d_model: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2, num_key_value_heads=2, hidden_size=d_model
        )
        self.model = SimpleNamespace(layers=nn.ModuleList([DistinctBasisLayer(d_model)]))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0](hidden)


def test_o_proj_capture_uses_true_runtime_input_not_layernorm():
    """Regression for the basis-mismatch bug: X must be the o_proj input."""
    model = DistinctBasisModel()
    tokens = (torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + 1.0) / 3.0

    with EngramContextCatcher(model, [0]) as catcher:
        model(tokens)
        inputs, outputs = catcher.get_activations()

    true_o_proj_input = model.model.layers[0].last_o_proj_input
    layernorm_out = tokens * 2.0
    assert not torch.allclose(true_o_proj_input, layernorm_out)

    np.testing.assert_allclose(inputs[0], centered(true_o_proj_input), atol=1e-6)
    np.testing.assert_allclose(outputs[0], centered(true_o_proj_input), atol=1e-6)
    # The captured input must NOT equal the layernorm output (the old bug).
    assert not np.allclose(inputs[0], centered(layernorm_out), atol=1e-3)

