from __future__ import annotations

from typing import Any, Callable
import numpy as np
from torch import nn

from .base import capture_tensor


class AttentionCatcher:
    """Specialized catcher for attention o_proj inputs and outputs."""

    def __init__(
        self,
        layer_index: int,
        input_layernorm: nn.Module | None,
        o_proj: nn.Module,
    ) -> None:
        self.layer_index = layer_index
        self.input_layernorm = input_layernorm
        self.o_proj = o_proj
        self.captured_input: np.ndarray | None = None
        self.captured_output: np.ndarray | None = None
        self.sink_input: np.ndarray | None = None
        self.sink_output: np.ndarray | None = None

    def make_input_hook(self) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            if self.captured_input is not None:
                raise RuntimeError(
                    f"decoder layer {self.layer_index} produced input_layernorm activations more than once; "
                    "EngramContextCatcher captures exactly one pass"
                )
            in_res = capture_tensor(output, self.layer_index, "input_layernorm", return_sink=True)
            if isinstance(in_res, tuple):
                self.captured_input, self.sink_input = in_res
            else:
                self.captured_input = in_res

        return hook

    def make_output_hook(self) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            if self.captured_output is not None:
                raise RuntimeError(
                    f"decoder layer {self.layer_index} produced self_attn.o_proj activations more than once; "
                    "EngramContextCatcher captures exactly one pass"
                )
            out_res = capture_tensor(output, self.layer_index, "o_proj_output", return_sink=True)
            if isinstance(out_res, tuple):
                self.captured_output, self.sink_output = out_res
            else:
                self.captured_output = out_res

        return hook

    def clear(self) -> None:
        self.captured_input = None
        self.captured_output = None
        self.sink_input = None
        self.sink_output = None
