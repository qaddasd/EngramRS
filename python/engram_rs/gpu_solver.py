"""Direct In-VRAM GPU Solver for low-latency LoRA compilation in EngramRS.

Eliminates GPU->CPU->Rust->Disk transfer by solving the ridge regression
and low-rank factorization entirely within PyTorch VRAM on GPU tensors.
For typical context lengths and model dimensions (e.g. d=4096, n=128),
the dual QR-accelerated solve runs in 10-20ms directly in VRAM.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch
from torch import nn


def _effective_lambda(lambda_reg: float, trace: torch.Tensor, d_in: int) -> float:
    """Relative ridge matching the Rust core (src/math/solvers.rs).

    lambda_eff = lambda_reg * trace(gram) / d_in, computed AFTER the null-space
    penalty. Keeps regularization meaningful at any activation scale and makes
    the GPU path numerically equivalent to the Rust primal/dual solvers.
    """
    trace_val = float(trace)
    if trace_val > 0.0 and math.isfinite(trace_val):
        return lambda_reg * trace_val / d_in
    return lambda_reg


def _validate_tensor_inputs(
    hidden_states: torch.Tensor,
    target_outputs: torch.Tensor,
    rank: int,
    lambda_reg: float,
    lambda_null: float,
    null_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    """Validate shapes, rank, finite values, and regularization hyperparameters."""
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states).__name__}")
    if not isinstance(target_outputs, torch.Tensor):
        raise TypeError(f"target_outputs must be a torch.Tensor, got {type(target_outputs).__name__}")

    # Flatten if 3D (batch, seq, d_model) -> (batch * seq, d_model)
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    y = target_outputs.reshape(-1, target_outputs.shape[-1]).contiguous()

    if x.ndim != 2 or x.numel() == 0:
        raise ValueError(f"hidden_states must be a non-empty 2D/3D tensor, got shape {hidden_states.shape}")
    if y.ndim != 2 or y.numel() == 0:
        raise ValueError(f"target_outputs must be a non-empty 2D/3D tensor, got shape {target_outputs.shape}")

    n, d = x.shape
    if y.shape != (n, d):
        raise ValueError(f"target_outputs shape {y.shape} does not match hidden_states shape {(n, d)}")

    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if rank > d:
        raise ValueError(f"rank ({rank}) cannot exceed d_model ({d})")

    if not isinstance(lambda_reg, (int, float)) or not math.isfinite(lambda_reg) or lambda_reg <= 0.0:
        raise ValueError(f"lambda_reg must be finite and positive, got {lambda_reg}")
    if not isinstance(lambda_null, (int, float)) or not math.isfinite(lambda_null) or lambda_null < 0.0:
        raise ValueError(f"lambda_null must be finite and non-negative, got {lambda_null}")

    x_null: torch.Tensor | None = None
    if null_states is not None:
        if not isinstance(null_states, torch.Tensor):
            raise TypeError(f"null_states must be a torch.Tensor, got {type(null_states).__name__}")
        x_null = null_states.reshape(-1, null_states.shape[-1]).contiguous()
        if x_null.shape[-1] != d:
            raise ValueError(
                f"null_states d_model ({x_null.shape[-1]}) does not match hidden_states d_model ({d})"
            )
        if x_null.numel() == 0:
            x_null = None

    if not torch.isfinite(x).all():
        raise ValueError("hidden_states contains non-finite values (NaN or Inf)")
    if not torch.isfinite(y).all():
        raise ValueError("target_outputs contains non-finite values (NaN or Inf)")
    if x_null is not None and not torch.isfinite(x_null).all():
        raise ValueError("null_states contains non-finite values (NaN or Inf)")

    return x, y, x_null, n, d


def _svd_anchors_gpu(
    x: torch.Tensor,
    y: torch.Tensor,
    max_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress x and y along the row dimension to at most `max_anchors` SVD anchors."""
    n, d = x.shape
    k = min(max_anchors, n, d)
    if k >= n:
        return x, y

    # X = U S V^T. Anchored X_a = S_k V_k^T (shape k x d), Y_a = U_k^T Y (shape k x d).
    # This preserves X^T X and X^T Y restricted to the rank-k subspace.
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    u_k = u[:, :k]
    s_k = s[:k]
    vh_k = vh[:k, :]

    anchored_x = s_k[:, None] * vh_k
    anchored_y = u_k.transpose(0, 1) @ y
    return anchored_x, anchored_y


def solve_layer_gpu(
    hidden_states: torch.Tensor,
    target_outputs: torch.Tensor,
    rank: int = 16,
    lambda_reg: float = 1e-3,
    lambda_null: float = 0.0,
    null_states: torch.Tensor | None = None,
    max_anchors: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve regularized least-squares for one layer and factorize into LoRA (A, B) in VRAM.

    Given inputs X (hidden_states) and targets Y (target_outputs), finds the weight matrix W
    minimizing:
        ||X W - Y||_F^2 + lambda_null * ||X_null W||_F^2 + lambda_reg * ||W||_F^2
    and factorizes W via SVD into:
        W ~= A^T B^T  where A has shape (rank, d_model), B has shape (d_model, rank)
    such that for inference in PEFT LoRA format:
        (x @ A^T) @ B^T = x @ W ~= y.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Activation input tensor of shape (tokens, d_model) or (batch, seq, d_model).
    target_outputs : torch.Tensor
        Activation target tensor of same shape as hidden_states.
    rank : int, default 16
        LoRA rank parameter (must be <= d_model).
    lambda_reg : float, default 1e-3
        L2 ridge regularization penalty.
    lambda_null : float, default 0.0
        Regularization penalty on null states.
    null_states : torch.Tensor | None, default None
        Optional unconditioned / background activations to regularize toward zero.
    max_anchors : int | None, default None
        Optional anchor count limit to compress long contexts via SVD anchors.
    device : str | torch.device | None, default None
        Target device for computation. If None, uses hidden_states.device.
    dtype : torch.dtype | None, default None
        Computation dtype. If None, defaults to torch.float32 for numerical stability.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (lora_a, lora_b) where lora_a has shape (rank, d_model) and lora_b has
        shape (d_model, rank), resident on the requested device.
    """
    orig_device = hidden_states.device
    target_device = torch.device(device) if device is not None else orig_device
    solve_dtype = dtype if dtype is not None else torch.float32

    # Move tensors to computation device and dtype
    x = hidden_states.to(device=target_device, dtype=solve_dtype)
    y = target_outputs.to(device=target_device, dtype=solve_dtype)
    x_null = (
        null_states.to(device=target_device, dtype=solve_dtype)
        if null_states is not None
        else None
    )

    x, y, x_null, n, d = _validate_tensor_inputs(
        x, y, rank, lambda_reg, lambda_null, x_null
    )

    # Apply SVD anchor compression if requested
    if max_anchors is not None and max_anchors > 0 and n > max_anchors:
        x, y = _svd_anchors_gpu(x, y, max_anchors)
        n = x.shape[0]

    null_n = x_null.shape[0] if (x_null is not None and lambda_null > 0.0) else 0
    total_samples = n + null_n

    eye_d = torch.eye(d, device=target_device, dtype=solve_dtype)

    # Dual formulation when total_samples < d (much faster when context tokens << d_model)
    if total_samples < d:
        p = total_samples
        if null_n > 0 and x_null is not None:
            scale_null = math.sqrt(lambda_null)
            z = torch.cat([x, scale_null * x_null], dim=0)
            rhs = torch.cat([y, torch.zeros((null_n, d), device=target_device, dtype=solve_dtype)], dim=0)
        else:
            z = x
            rhs = y

        # Dual Gram matrix: K = Z Z^T + lambda_eff * I_p  (shape p x p).
        # Relative ridge normalized by d_in (= d), NOT p, to match the Rust core.
        eye_p = torch.eye(p, device=target_device, dtype=solve_dtype)
        k_mat = z @ z.transpose(0, 1)
        lambda_eff = _effective_lambda(lambda_reg, k_mat.diagonal().sum(), d)
        k_mat = k_mat + lambda_eff * eye_p

        try:
            chol = torch.linalg.cholesky(k_mat)
            solved = torch.cholesky_solve(rhs, chol)
        except RuntimeError:
            # Fallback if Cholesky encounters near-singular conditioning
            k_mat = k_mat + 1e-5 * eye_p
            try:
                chol = torch.linalg.cholesky(k_mat)
                solved = torch.cholesky_solve(rhs, chol)
            except RuntimeError:
                solved = torch.linalg.solve(k_mat, rhs)

        # Acceleration: Thin QR on Z^T (d x p) gives Z^T = Q R with Q (d x p), R (p x p).
        # Then W = Z^T @ solved = Q @ (R @ solved).
        # We perform SVD on the small p x d matrix M = R @ solved:
        # M = U_M S V^T. Then U = Q @ U_M (d x p).
        # Avoids computing full (d x d) SVD!
        q, r = torch.linalg.qr(z.transpose(0, 1), mode="reduced")
        reduced = r @ solved  # shape (p, d)
        u_m, s, vh = torch.linalg.svd(reduced, full_matrices=False)
        u = q @ u_m  # shape (d, p)
        v = vh.transpose(0, 1)  # shape (d, p)
    else:
        # Primal: (X^T X + lambda_null X_null^T X_null + lambda_eff I) W = X^T Y.
        # Relative ridge computed AFTER the null penalty, matching the Rust core.
        gram = x.transpose(0, 1) @ x
        if null_n > 0 and x_null is not None:
            gram = gram + lambda_null * (x_null.transpose(0, 1) @ x_null)
        lambda_eff = _effective_lambda(lambda_reg, gram.diagonal().sum(), d)
        gram = gram + lambda_eff * eye_d
        cross = x.transpose(0, 1) @ y

        try:
            chol = torch.linalg.cholesky(gram)
            weights = torch.cholesky_solve(cross, chol)
        except RuntimeError:
            gram = gram + 1e-5 * eye_d
            try:
                chol = torch.linalg.cholesky(gram)
                weights = torch.cholesky_solve(cross, chol)
            except RuntimeError:
                weights = torch.linalg.solve(gram, cross)

        u, s, vh = torch.linalg.svd(weights, full_matrices=False)
        v = vh.transpose(0, 1)

    # Construct rank-truncated LoRA matrices
    components = min(rank, s.shape[0])
    s_trunc = torch.clamp(s[:components], min=0.0)
    scale = torch.sqrt(s_trunc)

    lora_a = torch.zeros((rank, d), device=target_device, dtype=solve_dtype)
    lora_b = torch.zeros((d, rank), device=target_device, dtype=solve_dtype)

    if components > 0:
        lora_a[:components, :] = scale[:, None] * u[:, :components].transpose(0, 1)
        lora_b[:, :components] = v[:, :components] * scale[None, :]

    return lora_a, lora_b


def solve_all_layers_gpu(
    hidden_states_list: Sequence[torch.Tensor],
    target_outputs_list: Sequence[torch.Tensor],
    rank: int = 16,
    lambda_reg: float = 1e-3,
    lambda_null: float = 0.0,
    null_states_list: Sequence[torch.Tensor | None] | None = None,
    layer_indices: Sequence[int] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve LoRA weights for all decoder layers entirely in VRAM.

    Parameters
    ----------
    hidden_states_list : Sequence[torch.Tensor]
        Input activations per layer.
    target_outputs_list : Sequence[torch.Tensor]
        Output activations per layer.
    rank : int, default 16
        LoRA rank.
    lambda_reg : float, default 1e-3
        Regularization penalty.
    lambda_null : float, default 0.0
        Null states penalty.
    null_states_list : Sequence[torch.Tensor | None] | None, default None
        Optional null states per layer.
    layer_indices : Sequence[int] | None, default None
        Indices of layers being compiled.
    device : str | torch.device | None, default None
        Target device for computation.
    dtype : torch.dtype | None, default None
        Computation dtype.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        List of (lora_a, lora_b) tuples for each layer.
    """
    num_layers = len(hidden_states_list)
    if len(target_outputs_list) != num_layers:
        raise ValueError(
            f"target_outputs_list length ({len(target_outputs_list)}) != "
            f"hidden_states_list length ({num_layers})"
        )
    if null_states_list is not None and len(null_states_list) != num_layers:
        raise ValueError(
            f"null_states_list length ({len(null_states_list)}) != "
            f"hidden_states_list length ({num_layers})"
        )

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(num_layers):
        x = hidden_states_list[idx]
        y = target_outputs_list[idx]
        x_null = null_states_list[idx] if null_states_list is not None else None
        lora_a, lora_b = solve_layer_gpu(
            hidden_states=x,
            target_outputs=y,
            rank=rank,
            lambda_reg=lambda_reg,
            lambda_null=lambda_null,
            null_states=x_null,
            device=device,
            dtype=dtype,
        )
        results.append((lora_a, lora_b))

    return results


def build_peft_state_dict(
    lora_weights: Sequence[tuple[torch.Tensor, torch.Tensor]],
    layer_indices: Sequence[int] | None = None,
    target_module: str = "self_attn.o_proj",
    base_prefix: str = "base_model.model.model.layers",
) -> dict[str, torch.Tensor]:
    """Convert a sequence of (lora_a, lora_b) into a PEFT-compatible state dictionary."""
    if layer_indices is None:
        layer_indices = list(range(len(lora_weights)))
    if len(layer_indices) != len(lora_weights):
        raise ValueError("layer_indices count must match lora_weights count")

    state_dict: dict[str, torch.Tensor] = {}
    for idx, (lora_a, lora_b) in zip(layer_indices, lora_weights):
        prefix = f"{base_prefix}.{idx}.{target_module}.lora_"
        state_dict[f"{prefix}A.weight"] = lora_a.contiguous()
        state_dict[f"{prefix}B.weight"] = lora_b.contiguous()

    return state_dict


class GPULayerCompiler:
    """High-level in-VRAM LoRA compiler for ultra-low latency inference adaptation.

    Runs entirely on GPU tensors, avoiding any CPU transfers or disk serialization.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int = 16,
        lambda_reg: float = 1e-3,
        lambda_null: float = 0.0,
        target_module: str = "self_attn.o_proj",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.lambda_null = lambda_null
        self.target_module = target_module
        self.device = device
        self.dtype = dtype

    def compile_activations(
        self,
        hidden_states_list: Sequence[torch.Tensor],
        attention_outputs_list: Sequence[torch.Tensor],
        null_states_list: Sequence[torch.Tensor | None] | None = None,
        layer_indices: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compile activation tensors directly into a PEFT state dictionary in VRAM."""
        lora_weights = solve_all_layers_gpu(
            hidden_states_list=hidden_states_list,
            target_outputs_list=attention_outputs_list,
            rank=self.rank,
            lambda_reg=self.lambda_reg,
            lambda_null=self.lambda_null,
            null_states_list=null_states_list,
            layer_indices=layer_indices,
            device=self.device,
            dtype=self.dtype,
        )
        return build_peft_state_dict(
            lora_weights,
            layer_indices=layer_indices,
            target_module=self.target_module,
        )

    def inject_peft_model(
        self,
        peft_model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """Inject compiled in-VRAM adapter weights into a PEFT model instance."""
        peft_model.load_state_dict(state_dict, strict=strict)
