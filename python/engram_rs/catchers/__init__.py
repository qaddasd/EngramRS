"""Modular activation catchers for HuggingFace-style decoder models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from .attention import AttentionCatcher
from .base import capture_tensor, find_decoder_layers
from .mlp import MLPCatcher


class EngramContextCatcher:
    """Capture sink-centered activations for decoder attention and MLP layers.

    Uses Rust SIMD kernels for zero-overhead PCIe transfer, BF16 unpacking,
    and sink-token centering.
    """

    SUPPORTED_TARGETS = ("o_proj", "down_proj")

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Iterable[int] | None = None,
        targets: Iterable[str] | None = None,
    ) -> None:
        self.model = model
        self._layers = find_decoder_layers(model)
        if layer_indices is None:
            layer_indices = range(len(self._layers))
        self.layer_indices = self._validate_layer_indices(layer_indices)

        for index in self.layer_indices:
            if index >= len(self._layers):
                raise IndexError(
                    f"decoder layer index {index} is out of range for "
                    f"{len(self._layers)} layers"
                )

        self.targets = self._resolve_targets(targets)
        self._attention_catchers: dict[int, AttentionCatcher] = {}
        self._mlp_catchers: dict[int, MLPCatcher] = {}

        for index in self.layer_indices:
            layer = self._layers[index]
            if "o_proj" in self.targets:
                input_layernorm = getattr(layer, "input_layernorm", None)
                self_attn = getattr(layer, "self_attn", None)
                o_proj = getattr(self_attn, "o_proj", None)
                if not isinstance(input_layernorm, nn.Module):
                    raise AttributeError(
                        f"decoder layer {index} ({type(layer).__name__}) has no "
                        "torch module named 'input_layernorm'"
                    )
                if not isinstance(o_proj, nn.Module):
                    raise AttributeError(
                        f"decoder layer {index} ({type(layer).__name__}) has no "
                        "torch module named 'self_attn.o_proj'"
                    )
                self._attention_catchers[index] = AttentionCatcher(
                    index, input_layernorm, o_proj
                )

            if "down_proj" in self.targets:
                mlp = getattr(layer, "mlp", None)
                down_proj = getattr(mlp, "down_proj", None)
                if not isinstance(down_proj, nn.Module):
                    raise AttributeError(
                        f"decoder layer {index} ({type(layer).__name__}) has no "
                        "torch module named 'mlp.down_proj'"
                    )
                self._mlp_catchers[index] = MLPCatcher(index, down_proj, layer=layer)

        self._handles: list[Any] = []
        self._inputs: dict[str, dict[int, np.ndarray]] = {t: {} for t in self.targets}
        self._outputs: dict[str, dict[int, np.ndarray]] = {t: {} for t in self.targets}
        self._cpu_tensors: dict[tuple[str, str, int], torch.Tensor] = {}

    def _resolve_targets(self, targets: Iterable[str] | None) -> tuple[str, ...]:
        if targets is None:
            first_layer = self._layers[self.layer_indices[0]]
            mlp = getattr(first_layer, "mlp", None)
            has_down_proj = isinstance(getattr(mlp, "down_proj", None), nn.Module)
            if has_down_proj:
                return ("o_proj", "down_proj")
            return ("o_proj",)

        if isinstance(targets, (str, bytes)):
            raise TypeError("targets must be an iterable of target module names")
        try:
            raw_targets = list(targets)
        except TypeError as error:
            raise TypeError("targets must be an iterable of target module names") from error

        resolved: list[str] = []
        for target in raw_targets:
            if not isinstance(target, str):
                raise TypeError(
                    f"target names must be strings, got {type(target).__name__}"
                )
            canonical = (
                "o_proj"
                if target in ("o_proj", "self_attn.o_proj")
                else "down_proj"
                if target in ("down_proj", "mlp.down_proj")
                else None
            )
            if canonical is None:
                raise ValueError(
                    f"unsupported target module {target!r}; supported targets are "
                    f"{self.SUPPORTED_TARGETS}"
                )
            if canonical not in resolved:
                resolved.append(canonical)
        if not resolved:
            raise ValueError("targets must contain at least one valid target module name")
        return tuple(resolved)

    @staticmethod
    def _validate_layer_indices(indices: Iterable[int]) -> tuple[int, ...]:
        if isinstance(indices, (str, bytes)):
            raise TypeError("layer_indices must be an iterable of non-negative integers")
        try:
            materialized = list(indices)
        except TypeError as error:
            raise TypeError("layer_indices must be an iterable of non-negative integers") from error

        seen: set[int] = set()
        validated: list[int] = []
        for index in materialized:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    f"layer indices must be non-negative integers, got {type(index).__name__}"
                )
            if index < 0:
                raise ValueError(f"layer indices must be non-negative, got {index}")
            if index in seen:
                raise ValueError(
                    f"layer index {index} appears more than once in layer_indices (duplicate)"
                )
            seen.add(index)
            validated.append(index)
        if not validated:
            raise ValueError("layer_indices must contain at least one layer index")
        return tuple(validated)

    @property
    def gqa_metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        query_heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", None) or query_heads
        hidden_size = getattr(config, "hidden_size", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and hidden_size is not None and query_heads is not None:
            if hidden_size % query_heads == 0:
                head_dim = hidden_size // query_heads
        groups = None
        if query_heads is not None and kv_heads is not None:
            if query_heads >= kv_heads and query_heads % kv_heads == 0:
                groups = query_heads // kv_heads
        return {
            "num_attention_heads": query_heads,
            "num_key_value_heads": kv_heads,
            "num_query_groups": groups,
            "head_dim": head_dim,
            "o_proj_output_repeated": False,
            "rope_inversion_applied": False,
            "rope_inversion_required": False,
        }

    @property
    def capture_metadata(self) -> dict[str, Any]:
        return {
            "layer_indices": self.layer_indices,
            "targets": list(self.targets),
            "layout": "[batch * sequence, d_features]",
            "sink_token_index": 0,
            "sink_centered_per_batch": True,
            **self.gqa_metadata,
        }

    def __enter__(self) -> "EngramContextCatcher":
        if self._handles:
            raise RuntimeError("EngramContextCatcher is already active")
        self._release_captures()
        try:
            for index in self.layer_indices:
                layer = self._layers[index]
                if "o_proj" in self.targets:
                    catcher = self._attention_catchers[index]
                    self._handles.append(
                        catcher.o_proj.register_forward_hook(
                            self._make_o_proj_hook(index, catcher)
                        )
                    )
                if "down_proj" in self.targets:
                    mlp_catcher = self._mlp_catchers[index]
                    self._handles.append(
                        mlp_catcher.down_proj.register_forward_hook(
                            self._make_down_proj_hook(index, mlp_catcher)
                        )
                    )
                    post_ln = getattr(layer, "post_attention_layernorm", None)
                    if isinstance(post_ln, nn.Module):
                        self._handles.append(
                            post_ln.register_forward_hook(
                                mlp_catcher.make_post_attention_layernorm_hook()
                            )
                        )
        except Exception:
            self.remove_hooks()
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.remove_hooks()
        return False

    def _make_o_proj_hook(self, layer_index: int, catcher: AttentionCatcher) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module
            if layer_index in self._outputs["o_proj"]:
                raise RuntimeError(
                    f"decoder layer {layer_index} produced self_attn.o_proj activations more than once; "
                    "EngramContextCatcher captures exactly one pass"
                )
            if not inputs:
                raise RuntimeError(
                    f"decoder layer {layer_index} self_attn.o_proj received an empty inputs tuple"
                )
            # Basis contract: X must be the true runtime input of o_proj
            # (post-attention multi-head concat), never the pre-attention layernorm.
            in_res = capture_tensor(inputs[0], layer_index, "o_proj_input", return_sink=True)
            if isinstance(in_res, tuple):
                in_arr, in_sink = in_res
                catcher.sink_input = in_sink
            else:
                in_arr = in_res
            self._inputs["o_proj"][layer_index] = in_arr

            out_res = capture_tensor(output, layer_index, "o_proj_output", return_sink=True)
            if isinstance(out_res, tuple):
                out_arr, out_sink = out_res
                catcher.sink_output = out_sink
            else:
                out_arr = out_res
            self._outputs["o_proj"][layer_index] = out_arr

        return hook

    def _make_down_proj_hook(self, layer_index: int, catcher: MLPCatcher) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module
            if layer_index in self._inputs["down_proj"]:
                raise RuntimeError(
                    f"decoder layer {layer_index} produced mlp.down_proj activations more than once; "
                    "EngramContextCatcher captures exactly one pass"
                )
            if not inputs:
                raise RuntimeError(
                    f"decoder layer {layer_index} mlp.down_proj received an empty inputs tuple"
                )
            in_res = capture_tensor(inputs[0], layer_index, "down_proj_input", return_sink=True)
            if isinstance(in_res, tuple):
                in_arr, in_sink = in_res
                catcher.sink_input = in_sink
            else:
                in_arr = in_res
            self._inputs["down_proj"][layer_index] = in_arr

            out_res = capture_tensor(output, layer_index, "down_proj_output", return_sink=True)
            if isinstance(out_res, tuple):
                out_arr, out_sink = out_res
                catcher.sink_output = out_sink
            else:
                out_arr = out_res
            self._outputs["down_proj"][layer_index] = out_arr

        return hook

    @staticmethod
    def _canonical_target(target: str) -> str:
        return "o_proj" if target in ("o_proj", "self_attn.o_proj") else "down_proj"

    def get_target_activations(self, target: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        canonical = self._canonical_target(target)
        if canonical not in self.targets:
            raise ValueError(
                f"target {target!r} was not captured; captured targets are {self.targets}"
            )

        in_map = self._inputs[canonical]
        out_map = self._outputs[canonical]
        if not in_map and not out_map:
            raise RuntimeError(
                "no activations were captured; run the model forward exactly once "
                "inside the EngramContextCatcher context"
            )

        missing_in = [idx for idx in self.layer_indices if idx not in in_map]
        missing_out = [idx for idx in self.layer_indices if idx not in out_map]
        if missing_in or missing_out:
            details = []
            if canonical == "o_proj":
                if missing_in:
                    details.append(f"missing self_attn.o_proj input captures for layers {missing_in}")
                if missing_out:
                    details.append(f"missing self_attn.o_proj captures for layers {missing_out}")
            else:
                if missing_in:
                    details.append(f"missing mlp.down_proj input captures for layers {missing_in}")
                if missing_out:
                    details.append(f"missing mlp.down_proj output captures for layers {missing_out}")
            raise RuntimeError("incomplete activation capture: " + "; ".join(details))

        inputs = [in_map[idx] for idx in self.layer_indices]
        outputs = [out_map[idx] for idx in self.layer_indices]
        for position, index in enumerate(self.layer_indices):
            in_shape = inputs[position].shape
            out_shape = outputs[position].shape
            if in_shape[0] != out_shape[0]:
                raise RuntimeError(
                    f"decoder layer {index} {canonical} token count mismatch: "
                    f"input {in_shape}, output {out_shape}"
                )
        return inputs, outputs

    def get_all_activations(self) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
        return {target: self.get_target_activations(target) for target in self.targets}

    def get_sink_tokens(self, target: str = "o_proj") -> dict[int, np.ndarray]:
        """Retrieve raw, uncentered sink tokens x0 per layer to guarantee null-space isolation."""
        canonical = self._canonical_target(target)
        sinks: dict[int, np.ndarray] = {}
        for idx in self.layer_indices:
            if canonical == "o_proj" and idx in self._attention_catchers:
                s = self._attention_catchers[idx].sink_input
                if s is not None:
                    sinks[idx] = s
            elif canonical == "down_proj" and idx in self._mlp_catchers:
                s = self._mlp_catchers[idx].sink_input
                if s is not None:
                    sinks[idx] = s
        return sinks

    def get_contrastive_target_activations(
        self,
    ) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
        """Get multi-target activations with Cross-Sublayer Contrastive Delta for down_proj.

        For mlp.down_proj, sets target output to the attention contextual shift Delta_Y_ctx
        and reconstructs test-time local input H_mlp_local to eliminate covariate shift.
        """
        acts = self.get_all_activations()
        if "down_proj" in acts and "o_proj" in acts:
            mlp_in, _ = acts["down_proj"]
            _, attn_out = acts["o_proj"]

            # Reconstruct covariate shift corrected local inputs H_mlp_local per layer
            corrected_mlp_in = []
            for pos, l_idx in enumerate(self.layer_indices):
                mlp_catcher = self._mlp_catchers.get(l_idx)
                delta_y = attn_out[pos]
                local_x = None
                if mlp_catcher is not None:
                    local_x = mlp_catcher.compute_local_input(delta_y)
                if local_x is not None:
                    corrected_mlp_in.append(local_x)
                else:
                    corrected_mlp_in.append(mlp_in[pos])

            # Set corrected local input and target output of down_proj to Delta_Y_ctx
            acts["down_proj"] = (corrected_mlp_in, attn_out)
        return acts

    def get_activations(self, target: str | None = None) -> Any:
        if target is not None:
            return self.get_target_activations(target)
        if "o_proj" in self.targets:
            return (self.hidden_states, self.attention_outputs)
        return self.get_target_activations(self.targets[0])

    @property
    def hidden_states(self) -> list[np.ndarray]:
        return self.get_target_activations("o_proj")[0]

    @property
    def attention_outputs(self) -> list[np.ndarray]:
        return self.get_target_activations("o_proj")[1]

    @property
    def mlp_inputs(self) -> list[np.ndarray]:
        return self.get_target_activations("down_proj")[0]

    @property
    def mlp_outputs(self) -> list[np.ndarray]:
        return self.get_target_activations("down_proj")[1]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _release_captures(self) -> None:
        for t in self.targets:
            self._inputs[t].clear()
            self._outputs[t].clear()
        self._cpu_tensors.clear()
        for attn_catcher in self._attention_catchers.values():
            attn_catcher.clear()
        for mlp_catcher in self._mlp_catchers.values():
            mlp_catcher.clear()

    def clear(self) -> None:
        self.remove_hooks()
        self._release_captures()

    def close(self) -> None:
        self.clear()


__all__ = [
    "AttentionCatcher",
    "EngramContextCatcher",
    "MLPCatcher",
    "capture_tensor",
    "find_decoder_layers",
]
