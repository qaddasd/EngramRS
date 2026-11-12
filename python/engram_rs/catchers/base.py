from __future__ import annotations

from typing import Any
import numpy as np
import torch

try:
    from ..engram_rs import center_and_unpack_bf16_py, center_f32_py
except ImportError:
    center_and_unpack_bf16_py = None
    center_f32_py = None


def find_decoder_layers(model: Any) -> Any:
    """Locate the sequence of decoder layers in a HuggingFace-style model."""
    paths = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
        ("layers",),
    )
    for path in paths:
        current: Any = model
        for component in path:
            current = getattr(current, component, None)
            if current is None:
                break
        if current is not None:
            try:
                len(current)
                current[0]
            except (TypeError, KeyError, IndexError):
                continue
            return current
    searched = ", ".join(".".join(path) for path in paths)
    raise AttributeError(
        "could not find a non-empty decoder layer sequence; searched " + searched
    )


def capture_tensor(
    raw_output: Any,
    layer_index: int,
    kind: str,
    return_sink: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Capture activations with Rust SIMD sink-centering and BF16 unpack.

    Args:
        raw_output: PyTorch tensor or tuple containing tensor from forward hook.
        layer_index: Index of the decoder layer.
        kind: Descriptive label for error messages (e.g. 'o_proj_input').
        return_sink: If True, also returns the raw uncentered sink token x0 [1, d_features].

    Returns:
        centered_array: Float32 NumPy array with sink token subtracted.
        (Optional) raw_sink: Float32 NumPy array of shape [1, d_features] if return_sink=True.
    """
    if isinstance(raw_output, (tuple, list)):
        if not raw_output:
            raise RuntimeError(
                f"decoder layer {layer_index} returned an empty {kind} output tuple"
            )
        raw_output = raw_output[0]
    if not isinstance(raw_output, torch.Tensor):
        raise TypeError(
            f"decoder layer {layer_index} {kind} output must be a tensor or "
            f"a tuple whose first item is a tensor, got {type(raw_output).__name__}"
        )

    tensor = raw_output
    with torch.no_grad():
        if tensor.ndim not in (2, 3):
            raise ValueError(
                f"decoder layer {layer_index} {kind} must have shape "
                "[tokens, features] or [batch, sequence, features], got "
                f"{tuple(tensor.shape)}"
            )
        if any(size == 0 for size in tensor.shape):
            raise ValueError(
                f"decoder layer {layer_index} {kind} has an empty "
                f"activation shape {tuple(tensor.shape)}"
            )

        # Extract raw uncentered sink token (first token of first batch)
        raw_sink: np.ndarray | None = None
        if return_sink:
            if tensor.ndim == 2:
                raw_sink_t = tensor[0:1, :]
            else:
                raw_sink_t = tensor[0:1, 0:1, :].reshape(1, -1)
            raw_sink = (
                raw_sink_t.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )

        # High-performance path: transfer in BF16 directly over PCIe, unpack & center in Rust SIMD
        if tensor.dtype == torch.bfloat16 and center_and_unpack_bf16_py is not None:
            cpu_bf16 = tensor.detach().to(device="cpu").contiguous()
            u16_np = cpu_bf16.view(torch.uint16).numpy()
            out_np = center_and_unpack_bf16_py(u16_np)
            matrix = out_np.reshape(-1, out_np.shape[-1])
            res = np.ascontiguousarray(matrix, dtype=np.float32)
            return (res, raw_sink) if return_sink and raw_sink is not None else res

        # High-performance path: float32 centered via Rust SIMD
        if center_f32_py is not None:
            cpu_f32 = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            out_np = center_f32_py(cpu_f32.numpy())
            matrix = out_np.reshape(-1, out_np.shape[-1])
            res = np.ascontiguousarray(matrix, dtype=np.float32)
            return (res, raw_sink) if return_sink and raw_sink is not None else res

        # Fallback path if Rust SIMD functions not bound
        cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if cpu.ndim == 2:
            sink = cpu[0:1, :]
        else:
            sink = cpu[:, 0:1, :]
        centered = (cpu - sink).contiguous()
        matrix = centered.reshape(-1, centered.shape[-1]).contiguous()
        array = matrix.numpy()
        if not array.flags.c_contiguous or array.dtype != np.float32:
            raise RuntimeError("failed to create contiguous CPU float32 capture")
        return (array, raw_sink) if return_sink and raw_sink is not None else array
