"""Fast Randomized SVD (Halko-Martinsson-Tropp 2011) with power iterations.

Supports both PyTorch tensors and NumPy ndarrays for low-rank approximation
and singular value decomposition of large covariance and weight matrices.
"""

from __future__ import annotations

from typing import Union
import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]


def randomized_svd(
    matrix: TensorLike,
    rank: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int | np.random.Generator | torch.Generator | None = None,
) -> tuple[TensorLike, TensorLike, TensorLike]:
    """Compute truncated SVD via randomized range finder with power iterations.

    Follows Algorithm 4.4 and 5.1 from Halko, Martinsson, & Tropp (2011):
    "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions".

    Args:
        matrix: 2D matrix of shape [m, n] (torch.Tensor or np.ndarray).
        rank: Target rank k (positive integer).
        n_oversamples: Number of additional random projections (p >= 0, default 10).
        n_iter: Number of power iterations (q >= 0, default 2 for rapid spectral decay).
        random_state: Optional seed or RNG generator.

    Returns:
        (u, s, vh):
            u: Left singular vectors of shape [m, rank].
            s: Singular values of shape [rank].
            vh: Right singular vectors transposed of shape [rank, n] (so v = vh.T).
    """
    if rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")
    if n_oversamples < 0:
        raise ValueError(f"n_oversamples must be non-negative, got {n_oversamples}")

    if isinstance(matrix, torch.Tensor):
        return _randomized_svd_torch(
            matrix, rank, n_oversamples, n_iter, random_state
        )
    if isinstance(matrix, np.ndarray):
        return _randomized_svd_numpy(
            matrix, rank, n_oversamples, n_iter, random_state
        )
    raise TypeError(f"matrix must be torch.Tensor or np.ndarray, got {type(matrix).__name__}")


def _randomized_svd_torch(
    a: torch.Tensor,
    rank: int,
    n_oversamples: int,
    n_iter: int,
    random_state: int | torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if a.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {tuple(a.shape)}")
    m, n = a.shape
    k = min(rank, m, n)
    if k == 0:
        empty_u = torch.empty((m, 0), dtype=a.dtype, device=a.device)
        empty_s = torch.empty((0,), dtype=a.dtype, device=a.device)
        empty_vh = torch.empty((0, n), dtype=a.dtype, device=a.device)
        return empty_u, empty_s, empty_vh

    # If matrix is small or rank equals min(m, n), full deterministic SVD is exact
    if min(m, n) <= k + n_oversamples:
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(a.to(torch.float32), full_matrices=False)
            u = u[:, :k].to(a.dtype)
            s = s[:k].to(a.dtype)
            vh = vh[:k, :].to(a.dtype)
            return u, s, vh

    l = min(m, n, k + n_oversamples)

    generator = None
    if isinstance(random_state, int):
        generator = torch.Generator(device=a.device if a.device.type != "mps" else "cpu")
        generator.manual_seed(random_state)
    elif isinstance(random_state, torch.Generator):
        generator = random_state

    with torch.no_grad():
        a_f32 = a.to(torch.float32)
        # Stage A: Randomized range finder with power iterations
        omega = torch.randn(
            (n, l),
            dtype=torch.float32,
            device=a.device,
            generator=generator,
        )

        y = torch.matmul(a_f32, omega)
        q, _ = torch.linalg.qr(y, mode="reduced")

        for _ in range(n_iter):
            # Power iteration with QR re-orthogonalization to avoid numerical instability
            z = torch.matmul(a_f32.transpose(0, 1), q)
            q_z, _ = torch.linalg.qr(z, mode="reduced")
            y = torch.matmul(a_f32, q_z)
            q, _ = torch.linalg.qr(y, mode="reduced")

        # Stage B: Project and compute SVD of small matrix B (l x n)
        b = torch.matmul(q.transpose(0, 1), a_f32)
        u_b, s, vh = torch.linalg.svd(b, full_matrices=False)

        u = torch.matmul(q, u_b)

        u = u[:, :k].to(a.dtype)
        s = s[:k].to(a.dtype)
        vh = vh[:k, :].to(a.dtype)

    return u, s, vh


def _randomized_svd_numpy(
    a: np.ndarray,
    rank: int,
    n_oversamples: int,
    n_iter: int,
    random_state: int | np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if a.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {a.shape}")
    m, n = a.shape
    k = min(rank, m, n)
    if k == 0:
        return (
            np.empty((m, 0), dtype=a.dtype),
            np.empty((0,), dtype=a.dtype),
            np.empty((0, n), dtype=a.dtype),
        )

    if min(m, n) <= k + n_oversamples:
        u, s, vh = np.linalg.svd(a.astype(np.float32, copy=False), full_matrices=False)
        return (
            u[:, :k].astype(a.dtype, copy=False),
            s[:k].astype(a.dtype, copy=False),
            vh[:k, :].astype(a.dtype, copy=False),
        )

    l = min(m, n, k + n_oversamples)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(int(random_state))
    else:
        rng = np.random.default_rng()

    a_f32 = a.astype(np.float32, copy=False)
    omega = rng.standard_normal((n, l), dtype=np.float32)

    y = a_f32 @ omega
    q, _ = np.linalg.qr(y, mode="reduced")

    for _ in range(n_iter):
        z = a_f32.T @ q
        q_z, _ = np.linalg.qr(z, mode="reduced")
        y = a_f32 @ q_z
        q, _ = np.linalg.qr(y, mode="reduced")

    b = q.T @ a_f32
    u_b, s, vh = np.linalg.svd(b, full_matrices=False)

    u = q @ u_b

    return (
        u[:, :k].astype(a.dtype, copy=False),
        s[:k].astype(a.dtype, copy=False),
        vh[:k, :].astype(a.dtype, copy=False),
    )
