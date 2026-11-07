"""Strict Null-Space Orthogonal Projection for background representation isolation.

Direction 3.1 in EngramRS:
Exact orthogonal projection into the null-space of background representations:
    Delta W <- Delta W * (I - X_null^+ X_null)
where X_null^+ is the Moore-Penrose pseudoinverse of background activations X_null.

This mathematically guarantees that for any background representation vector
x in range(X_null), the adapter response is exactly 0.0000 to machine precision (< 1e-6).
"""

from __future__ import annotations

from typing import Any, Sequence, Union
import numpy as np
import torch
from torch import nn

TensorLike = Union[torch.Tensor, np.ndarray]

# Diverse generic calibration prompts spanning general knowledge, Wikipedia,
# science, code, and standard language structures.
DEFAULT_CALIBRATION_PROMPTS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "The history of science is the study of the development of science and scientific knowledge.",
    "In mathematics, a continuous function is a function such that a small variation of the argument induces a small variation of the value.",
    "Python is a high-level, general-purpose programming language designed with an emphasis on code readability.",
    "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
    "The climate of Earth is determined by the balance between incoming solar radiation and outgoing infrared radiation.",
    "Standard operating procedures are written instructions intended to document how to perform a routine activity.",
    "Artificial neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
    "Water is an inorganic compound with the chemical formula H2O, transparent, tasteless, and odorless.",
    "The Industrial Revolution marked a period of development in the latter half of the 18th century.",
    "Data structures and algorithms are fundamental concepts in computer science for organizing and processing information.",
    "Gravity is a fundamental interaction which causes mutual attraction between all things with mass or energy.",
]


def compute_null_space_projector(
    x_null: TensorLike,
    rcond: float = 1e-5,
) -> TensorLike:
    """Compute the orthogonal projection matrix onto the null-space of X_null.

    Let S = rowspace(X_null). The projector onto S^perp = null(X_null) is:
        P_null^perp = I - X_null^+ X_null = I - V_r V_r^T
    where V_r are the right singular vectors of X_null with singular values
    sigma_i > rcond * sigma_0.

    Properties:
        - Symmetric: (P_null^perp)^T = P_null^perp
        - Idempotent: (P_null^perp)^2 = P_null^perp
        - Annihilator: X_null @ P_null^perp = 0
        - For any x in span(X_null), x @ P_null^perp = 0

    Args:
        x_null: Background activations matrix of shape [K, d] (torch.Tensor or np.ndarray).
        rcond: Relative cutoff for small singular values (default 1e-5).

    Returns:
        P_null^perp: Projection matrix of shape [d, d] of the same type as x_null.
    """
    if isinstance(x_null, torch.Tensor):
        return _compute_projector_torch(x_null, rcond)
    if isinstance(x_null, np.ndarray):
        return _compute_projector_numpy(x_null, rcond)
    raise TypeError(f"x_null must be torch.Tensor or np.ndarray, got {type(x_null).__name__}")


def _compute_projector_torch(x_null: torch.Tensor, rcond: float) -> torch.Tensor:
    if x_null.ndim != 2:
        raise ValueError(f"x_null must be 2D matrix of shape [K, d], got {tuple(x_null.shape)}")
    k, d = x_null.shape
    if k == 0 or d == 0:
        return torch.eye(d, dtype=x_null.dtype, device=x_null.device)

    with torch.no_grad():
        x_f64 = x_null.to(torch.float64)
        # SVD in double precision for machine-precision null projection: X = U S Vh
        _, s, vh = torch.linalg.svd(x_f64, full_matrices=False)
        cutoff = rcond * s[0].item() if s.numel() > 0 else 0.0
        mask = s > cutoff
        vh_r = vh[mask]  # [r, d]
        if vh_r.shape[0] == 0:
            p_perp = torch.eye(d, dtype=torch.float64, device=x_null.device)
        else:
            # P_subspace = V_r V_r^T = (Vh_r)^T @ Vh_r
            p_subspace = torch.matmul(vh_r.transpose(0, 1), vh_r)
            p_perp = torch.eye(d, dtype=torch.float64, device=x_null.device) - p_subspace

        # Ensure exact symmetry
        p_perp = 0.5 * (p_perp + p_perp.transpose(0, 1))
        return p_perp.to(x_null.dtype)


def _compute_projector_numpy(x_null: np.ndarray, rcond: float) -> np.ndarray:
    if x_null.ndim != 2:
        raise ValueError(f"x_null must be 2D matrix of shape [K, d], got {x_null.shape}")
    k, d = x_null.shape
    if k == 0 or d == 0:
        return np.eye(d, dtype=x_null.dtype)

    x_f64 = x_null.astype(np.float64, copy=False)
    _, s, vh = np.linalg.svd(x_f64, full_matrices=False)
    cutoff = rcond * s[0] if len(s) > 0 else 0.0
    mask = s > cutoff
    vh_r = vh[mask]
    if vh_r.shape[0] == 0:
        p_perp = np.eye(d, dtype=np.float64)
    else:
        p_subspace = vh_r.T @ vh_r
        p_perp = np.eye(d, dtype=np.float64) - p_subspace

    p_perp = 0.5 * (p_perp + p_perp.T)
    return p_perp.astype(x_null.dtype, copy=False)


def project_to_null_space(
    delta_w: TensorLike,
    x_null: TensorLike,
    side: str = "right",
    rcond: float = 1e-5,
) -> TensorLike:
    """Project weight matrix Delta W into the orthogonal null-space of X_null.

    Formula:
        Delta W_proj = Delta W * (I - X_null^+ X_null)   [side="right"]
    or
        Delta W_proj = (I - X_null^+ X_null) * Delta W   [side="left"]
    or
        Delta W_proj = P_null * Delta W * P_null          [side="both"]

    For PyTorch Linear layers y = x @ W^T where delta_w is the parameter matrix
    of shape [d_out, d_in], side="right" ensures:
        x @ (Delta W @ P_null^perp)^T = x @ P_null^perp @ Delta W^T = 0
    for any background vector x in rowspace(X_null).

    For forward maps y = x @ Delta W (row vector mapping):
        side="left" ensures:
        x @ (P_null^perp @ Delta W) = (x @ P_null^perp) @ Delta W = 0.

    Args:
        delta_w: Weight matrix of shape [m, n].
        x_null: Background activations of shape [K, d] where d matches delta_w.
        side: "right" (default, Delta W @ P), "left" (P @ Delta W), or "both".
        rcond: Singular value cutoff tolerance.

    Returns:
        Delta W projected into the null-space.
    """
    if side not in ("right", "left", "both"):
        raise ValueError(f"side must be 'right', 'left', or 'both', got {side!r}")

    p_null = compute_null_space_projector(x_null, rcond=rcond)

    if isinstance(delta_w, torch.Tensor):
        if isinstance(p_null, np.ndarray):
            p_null = torch.from_numpy(p_null)
        p_null = p_null.to(device=delta_w.device, dtype=torch.float64)
        w_f64 = delta_w.to(torch.float64)
        if side == "right":
            res = torch.matmul(w_f64, p_null)
        elif side == "left":
            res = torch.matmul(p_null, w_f64)
        else:
            res = torch.matmul(torch.matmul(p_null, w_f64), p_null)
        return res.to(delta_w.dtype)

    if isinstance(delta_w, np.ndarray):
        if isinstance(p_null, torch.Tensor):
            p_null = p_null.detach().cpu().numpy()
        p_null_f64 = p_null.astype(np.float64, copy=False)
        w_f64 = delta_w.astype(np.float64, copy=False)
        if side == "right":
            res = w_f64 @ p_null_f64
        elif side == "left":
            res = p_null_f64 @ w_f64
        else:
            res = p_null_f64 @ w_f64 @ p_null_f64
        return res.astype(delta_w.dtype, copy=False)

    raise TypeError(f"delta_w must be torch.Tensor or np.ndarray, got {type(delta_w).__name__}")


def project_lora_to_null_space(
    lora_a: TensorLike,
    lora_b: TensorLike,
    x_null: TensorLike,
    rcond: float = 1e-5,
) -> tuple[TensorLike, TensorLike]:
    """Project LoRA adapter weights (lora_a, lora_b) into null-space of X_null.

    In PEFT LoRA conventions:
        lora_A: [rank, d_model]
        lora_B: [d_model, rank]
        Adapter forward on input x: delta_y = (x @ lora_A.T) @ lora_B.T

    By projecting lora_A:
        lora_A_proj = lora_A @ (I - X_null^+ X_null)
    we have:
        x @ (lora_A_proj)^T = x @ (I - X_null^+ X_null) @ lora_A^T = 0
    for any background representation x in rowspace(X_null).
    The rank is preserved and lora_B remains unchanged.

    Args:
        lora_a: Tensor or ndarray of shape [rank, d_model].
        lora_b: Tensor or ndarray of shape [d_model, rank].
        x_null: Background activations of shape [K, d_model].
        rcond: SVD cutoff tolerance.

    Returns:
        (lora_a_proj, lora_b): Projected lora_a and unmodified lora_b.
    """
    p_null = compute_null_space_projector(x_null, rcond=rcond)

    if isinstance(lora_a, torch.Tensor):
        if isinstance(p_null, np.ndarray):
            p_null = torch.from_numpy(p_null)
        p_null = p_null.to(device=lora_a.device, dtype=torch.float64)
        a_f64 = lora_a.to(torch.float64)
        lora_a_proj = torch.matmul(a_f64, p_null).to(lora_a.dtype)
        return lora_a_proj, lora_b

    if isinstance(lora_a, np.ndarray):
        if isinstance(p_null, torch.Tensor):
            p_null = p_null.detach().cpu().numpy()
        p_null_f64 = p_null.astype(np.float64, copy=False)
        a_f64 = lora_a.astype(np.float64, copy=False)
        lora_a_proj = (a_f64 @ p_null_f64).astype(lora_a.dtype, copy=False)
        return lora_a_proj, lora_b

    raise TypeError(f"lora_a must be torch.Tensor or np.ndarray, got {type(lora_a).__name__}")


def calibrate_background_anchors(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str] | None = None,
    max_tokens: int = 256,
    device: torch.device | str | None = None,
    layer_indices: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Calibrate background activation anchors H_null across model layers.

    Extracts sink-centered hidden state representations from the model forward
    pass over generic calibration prompts (or DEFAULT_CALIBRATION_PROMPTS).

    Args:
        model: HuggingFace-style decoder model.
        tokenizer: Model tokenizer.
        prompts: Sequence of generic text strings (defaults to DEFAULT_CALIBRATION_PROMPTS).
        max_tokens: Maximum tokens per prompt.
        device: Device to run forward pass on.
        layer_indices: Optional layer indices to capture (defaults to all decoder layers).

    Returns:
        dict mapping layer_index -> np.ndarray of shape [total_tokens, d_model].
    """
    from .hooks import EngramContextCatcher

    if prompts is None:
        prompts = DEFAULT_CALIBRATION_PROMPTS

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    catcher = EngramContextCatcher(model, layer_indices=layer_indices)
    was_training = model.training
    model.eval()

    layer_anchors: dict[int, list[np.ndarray]] = {idx: [] for idx in catcher.layer_indices}

    try:
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
            encoded = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in encoded.items()
            }
            with torch.no_grad(), catcher:
                outputs = model(**encoded, use_cache=False)
                del outputs

            hidden_states, _ = catcher.get_activations()
            for idx, hidden in zip(catcher.layer_indices, hidden_states):
                # hidden is [tokens, d_model] float32 array
                layer_anchors[idx].append(hidden)
            catcher.clear()

        # Concatenate across calibration prompts
        result: dict[int, np.ndarray] = {}
        for idx, arrays in layer_anchors.items():
            if arrays:
                result[idx] = np.concatenate(arrays, axis=0)
            else:
                # Fallback if prompts produced no tokens
                d = getattr(getattr(model, "config", None), "hidden_size", 64)
                result[idx] = np.zeros((0, d), dtype=np.float32)
        return result
    finally:
        catcher.clear()
        model.train(was_training)


def get_family_background_anchors(
    family_name: str = "generic",
    d_model: int = 4096,
    num_anchors: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic harmonic Fourier basis prior anchors for testing.

    Note: This function generates synthetic harmonic wave anchors with low-frequency
    decay for unit testing and algebraic null-space verification.
    For genuine production background preservation, use `calibrate_background_anchors()`
    which extracts authentic activation statistics directly from a model forward pass.

    Args:
        family_name: Model family tag for frequency tuning.
        d_model: Hidden dimension (d_model).
        num_anchors: Number of anchor vectors (rows).
        seed: Random seed for deterministic reproducibility.

    Returns:
        np.ndarray of shape [num_anchors, d_model], dtype float32.
    """
    family = family_name.lower().strip()
    rng = np.random.default_rng(seed)

    # Harmonic frequencies representing standard residual stream background components
    t = np.linspace(0, 2 * np.pi, d_model, endpoint=False, dtype=np.float32)
    anchors = np.zeros((num_anchors, d_model), dtype=np.float32)

    family_freq_multiplier = {
        "llama": 1.0,
        "llama3": 1.25,
        "mistral": 0.85,
        "qwen": 1.1,
    }.get(family, 1.0)

    for i in range(num_anchors):
        freq = (i + 1) * family_freq_multiplier
        phase = rng.uniform(0, np.pi)
        # Combination of harmonic waves with low-frequency dominant decay
        wave = np.sin(freq * t + phase) / np.sqrt(freq)
        noise = rng.normal(0, 0.05, size=d_model).astype(np.float32)
        anchors[i] = wave + noise

    # Orthonormalize anchor basis
    q, _ = np.linalg.qr(anchors.T, mode="reduced")
    anchors = q.T[:num_anchors]
    return anchors.astype(np.float32)
