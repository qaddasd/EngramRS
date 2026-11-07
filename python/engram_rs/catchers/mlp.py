from __future__ import annotations

from typing import Any, Callable
import numpy as np
import torch
from torch import nn

from .base import capture_tensor


class MLPCatcher:
    """Specialized catcher for MLP intermediate activations (SwiGLU -> down_proj)."""

    def __init__(
        self,
        layer_index: int,
        down_proj: nn.Module,
        layer: nn.Module | None = None,
    ) -> None:
        self.layer_index = layer_index
        self.down_proj = down_proj
        self.layer = layer
        self.captured_input: np.ndarray | None = None
        self.captured_output: np.ndarray | None = None
        self.sink_input: np.ndarray | None = None
        self.sink_output: np.ndarray | None = None
        self.override_target_output: np.ndarray | None = None
        self.raw_mid_input: torch.Tensor | None = None

    def make_post_attention_layernorm_hook(self) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module, output
            if inputs:
                t = inputs[0]
                if isinstance(t, torch.Tensor):
                    self.raw_mid_input = t.detach()

        return hook

    def compute_local_input(self, delta_y: np.ndarray | torch.Tensor) -> np.ndarray | None:
        """Compute Covariate Shift Corrected MLP input H_mlp_local.

        Reconstructs test-time local input h_mid_local = h_mid_full - delta_Y_ctx
        and feeds it through post_attention_layernorm and SwiGLU (gate_proj & up_proj)
        so that down_proj is trained on exact test-time inference representations.
        """
        if self.layer is None or self.raw_mid_input is None:
            return None

        post_ln = getattr(self.layer, "post_attention_layernorm", None)
        mlp = getattr(self.layer, "mlp", None)
        if post_ln is None or mlp is None:
            return None

        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        if gate_proj is None or up_proj is None:
            return None

        device = self.raw_mid_input.device
        dtype = self.raw_mid_input.dtype

        # Convert delta_y to tensor matching raw_mid_input shape
        if isinstance(delta_y, np.ndarray):
            delta_tensor = torch.from_numpy(delta_y).to(device=device, dtype=dtype)
        else:
            delta_tensor = delta_y.to(device=device, dtype=dtype)

        if delta_tensor.ndim == 2 and self.raw_mid_input.ndim == 3:
            delta_tensor = delta_tensor.view(self.raw_mid_input.shape)

        try:
            with torch.no_grad():
                h_mid_local = self.raw_mid_input - delta_tensor
                normed = post_ln(h_mid_local)
                act_fn = getattr(mlp, "act_fn", None)
                if act_fn is None:
                    import torch.nn.functional as F
                    act_fn = F.silu
                local_h = act_fn(gate_proj(normed)) * up_proj(normed)
                res = capture_tensor(local_h, self.layer_index, "down_proj_local_input")
                if isinstance(res, tuple):
                    return res[0]
                return res
        except Exception:
            return None

    def make_hook(self) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del module
            if self.captured_input is not None or self.captured_output is not None:
                raise RuntimeError(
                    f"decoder layer {self.layer_index} produced mlp.down_proj activations more than once; "
                    "EngramContextCatcher captures exactly one pass"
                )
            if not inputs:
                raise RuntimeError(
                    f"decoder layer {self.layer_index} mlp.down_proj received an empty inputs tuple"
                )
            in_tensor = inputs[0]
            in_res = capture_tensor(in_tensor, self.layer_index, "down_proj_input", return_sink=True)
            if isinstance(in_res, tuple):
                self.captured_input, self.sink_input = in_res
            else:
                self.captured_input = in_res

            out_res = capture_tensor(output, self.layer_index, "down_proj_output", return_sink=True)
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
        self.override_target_output = None
        self.raw_mid_input = None
