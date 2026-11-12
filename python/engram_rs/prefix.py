"""RoPE-correct virtual KV-cache prefix compilation for EngramRS.

The residual-KV tier guarantees verbatim recall by keeping *exact* context tokens
in a short resident prefix. Two correctness rules drive this module (REDESIGN_V2
§2.4, fixes C6/M6):

1. **Select K, never merge.** RoPE bakes a position-dependent phase into every
   cached key. A linear blend of keys from different positions has no coherent
   phase, so spectral *merging* of post-RoPE keys collapses attention. We instead
   *select* the top-B tokens by importance (SnapKV-style) and keep their exact
   key/value pairs, selected at the *same* indices so each attention weight still
   multiplies its own value.

2. **Re-RoPE to contiguous positions.** Selected tokens originally lived at
   scattered positions p_orig. We invert RoPE (K_pre = R(-p_orig) K), then re-apply
   it at the new contiguous slots (K_load = R(p_new) K_pre, p_new = 0..B-1). The
   prefix then occupies positions 0..B like an ordinary short prefix, so
   `cache_position = arange(B, B+q)` is phase-consistent and native prefix-caching
   in vLLM/SGLang works out of the box.

The RoPE math mirrors HuggingFace's NeoX half-split convention and the Rust core
(`src/rope_gqa.rs`, exported as `rope_apply_py` / `rope_inverse_py`). The Rust path
is used for CPU float32 tensors when the native extension is available; a torch
reference (device-agnostic, GPU-ready) is the fallback and the test oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

try:  # Rust core acceleration (optional; torch reference is the fallback).
    from .engram_rs import rope_apply_py as _rust_rope_apply
    from .engram_rs import rope_inverse_py as _rust_rope_inverse

    _HAS_RUST_ROPE = True
except Exception:  # pragma: no cover - native extension absent or stale
    _HAS_RUST_ROPE = False


CompressionMethod = Literal[
    "select", "svd", "spectral", "joint_svd", "energy", "nystrom"
]


def _rope_tables(
    positions: torch.Tensor,
    head_dim: int,
    base: float,
    partial_rotary_factor: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-token cos/sin for NeoX half-split RoPE, shape [T, rot_dim/2]."""
    rot_dim = int(head_dim * partial_rotary_factor)
    rot_dim -= rot_dim % 2
    if rot_dim < 2:
        raise ValueError(
            f"partial_rotary_factor {partial_rotary_factor} leaves rot_dim={rot_dim}; "
            "need at least 2 rotated dimensions"
        )
    half = rot_dim // 2
    inv_freq = base ** (
        -torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim
    )
    pos = positions.to(device=device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype), rot_dim


def _rope_rotate_torch(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    partial_rotary_factor: float,
    inverse: bool,
) -> torch.Tensor:
    """Torch reference RoPE on x[..., T, D]; positions[T] shared across leading dims."""
    cos, sin, rot_dim = _rope_tables(
        positions, x.shape[-1], base, partial_rotary_factor, x.device, x.dtype
    )
    half = rot_dim // 2
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    if inverse:
        r1 = x1 * cos + x2 * sin
        r2 = x2 * cos - x1 * sin
    else:
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
    return torch.cat([r1, r2, x_pass], dim=-1)


def _rope_transform_kv(
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float,
    partial_rotary_factor: float,
    inverse: bool,
) -> torch.Tensor:
    """Apply/invert RoPE on key[B, H, T, D] with shared per-token positions[T].

    Uses the Rust core for CPU float32 tensors when available; otherwise the torch
    reference. Both implement the identical NeoX half-split rotation.
    """
    if key.ndim != 4:
        raise ValueError(f"key must be 4D [B, H, T, D], got shape {tuple(key.shape)}")
    if positions.ndim != 1 or positions.shape[0] != key.shape[-2]:
        raise ValueError(
            f"positions must be 1D of length T={key.shape[-2]}, got {tuple(positions.shape)}"
        )

    if _HAS_RUST_ROPE and key.device.type == "cpu" and key.dtype == torch.float32:
        b, h, t, d = key.shape
        x_np = (
            key.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(b * t, h, d)
            .detach()
            .cpu()
            .numpy()
        )
        pos_np = positions.detach().cpu().to(torch.int64).numpy().repeat(b)
        fn = _rust_rope_inverse if inverse else _rust_rope_apply
        out = fn(x_np, pos_np, float(base), float(partial_rotary_factor))
        return (
            torch.from_numpy(np.ascontiguousarray(out))
            .reshape(b, t, h, d)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    return _rope_rotate_torch(key, positions, base, partial_rotary_factor, inverse)


def rope_inverse_kv(
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    partial_rotary_factor: float = 1.0,
) -> torch.Tensor:
    """Undo RoPE: K_pre = R(-positions) K, for key[B, H, T, D]."""
    return _rope_transform_kv(key, positions, base, partial_rotary_factor, inverse=True)


def rope_apply_kv(
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
    partial_rotary_factor: float = 1.0,
) -> torch.Tensor:
    """Apply RoPE: K_load = R(positions) K_pre, for key[B, H, T, D]."""
    return _rope_transform_kv(key, positions, base, partial_rotary_factor, inverse=False)


def rope_config_from_model(model: nn.Module) -> tuple[float, float] | None:
    """Extract (rope_theta, partial_rotary_factor) from a HF model config.

    Returns None for models without rotary embeddings (RoPE transform is skipped).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    base = getattr(cfg, "rope_theta", None)
    if base is None:
        return None
    factor = getattr(cfg, "partial_rotary_factor", 1.0) or 1.0
    return float(base), float(factor)


def _token_importance(key: torch.Tensor) -> torch.Tensor:
    """Per-token leverage importance ||k||^2, aggregated over batch and heads -> [T].

    RoPE is a rotation, so ||R(p) k|| = ||k||: leverage is identical pre/post RoPE,
    which lets us score post-RoPE cache keys directly. A single shared index set is
    used across heads so the HF cache keeps a uniform sequence length.
    """
    leverage = (key * key).sum(dim=-1)  # [B, H, T]
    return leverage.mean(dim=tuple(range(leverage.ndim - 1)))  # [T]


def snapkv_importance(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    obs_window: int = 32,
    query_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-layer, per-token importance for SnapKV selection.

    Two regimes:

    * Query-aware (``query_ids`` given; RECOMMENDED for verbatim recall). The
      observation window is the *question* itself: we forward [context, query]
      and score each context token by the attention it receives from the query
      tokens. This surfaces exactly the fact-bearing tokens the question needs,
      which is what makes needle recall possible at high compression.

    * Query-agnostic (``query_ids`` None). The window is the context's own last
      ``obs_window`` tokens — a fixed, reusable proxy for "what this context
      attends to". It is LOSSY: needles buried far from the context tail receive
      little attention from that window and may be dropped, so it cannot
      guarantee verbatim multi-needle recall (verified: 0/5 on a 2k-token,
      5-needle haystack at a 256-token budget). Use query-aware for facts.

    Returns a tensor of shape (num_layers, context_seq_len). Requires eager
    attention (``output_attentions=True``); raises if the model returns no
    attentions (e.g. SDPA/flash), since silently falling back to leverage would
    hide recall loss.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D (batch, seq), got {tuple(input_ids.shape)}")
    ctx_len = input_ids.shape[1]
    if obs_window <= 0:
        raise ValueError(f"obs_window must be positive, got {obs_window}")

    if query_ids is not None:
        if query_ids.ndim != 2 or query_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("query_ids must be 2D with the same batch size as input_ids")
        q_len = query_ids.shape[1]
        if q_len == 0:
            raise ValueError("query_ids must be non-empty for query-aware selection")
        query_ids = query_ids.to(input_ids.device)
        combined = torch.cat([input_ids, query_ids], dim=1)
        if attention_mask is not None:
            combined_mask = torch.cat(
                [attention_mask, torch.ones_like(query_ids, dtype=attention_mask.dtype)], dim=1
            )
        else:
            combined_mask = None
        window = q_len
    else:
        combined = input_ids
        combined_mask = attention_mask
        window = min(obs_window, ctx_len)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=combined,
                attention_mask=combined_mask,
                use_cache=False,
                output_attentions=True,
            )
    finally:
        model.train(was_training)

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError(
            "model returned no attention weights; load it with "
            "attn_implementation='eager' to use SnapKV selection"
        )

    device = input_ids.device
    per_layer: list[torch.Tensor] = []
    for attn in attentions:
        # attn: (B, Hq, T_total, T_total). The window is the last `window`
        # positions (query tokens when query-aware, context tail otherwise);
        # score only the context tokens [0, ctx_len).
        window_attn = attn[:, :, -window:, :ctx_len].float()  # (B, Hq, w, ctx_len)
        scores = window_attn.sum(dim=-2)  # (B, Hq, ctx_len)
        scores = scores.mean(dim=1)  # (B, ctx_len) average over query heads
        scores = scores.mean(dim=0)  # (ctx_len,) average over batch
        per_layer.append(scores.to(device))
    return torch.stack(per_layer, dim=0)  # (L, ctx_len)


def select_compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_len: int = 32,
    preserve_sinks: int = 1,
    scores: torch.Tensor | None = None,
    pad_to_prefix_len: bool = True,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the top-`prefix_len` exact tokens (RoPE-safe; keys are never merged).

    Parameters
    ----------
    key, value : torch.Tensor
        Shape (..., seq_len, head_dim), typically (B, num_kv_heads, T, D).
    prefix_len : int
        Target number of resident prefix tokens B.
    preserve_sinks : int
        Number of initial sink tokens kept verbatim before the selected tail.
    scores : torch.Tensor | None
        Optional per-token importance of shape (seq_len,). Defaults to key leverage.
    pad_to_prefix_len : bool
        Zero-pad when fewer than prefix_len tokens are available.
    return_indices : bool
        Also return the selected original positions (length prefix_len when padded).

    Returns
    -------
    (k_sel, v_sel) or (k_sel, v_sel, indices)
        Selected tensors of shape (..., prefix_len, head_dim).
    """
    if key.shape != value.shape:
        raise ValueError(f"key shape {key.shape} does not match value shape {value.shape}")
    if key.ndim < 2:
        raise ValueError(f"key and value must have at least 2 dimensions, got {key.shape}")
    if prefix_len <= 0:
        raise ValueError(f"prefix_len must be positive, got {prefix_len}")

    seq_len = key.shape[-2]
    head_dim = key.shape[-1]
    leading_dims = key.shape[:-2]
    device = key.device

    if scores is None:
        importance = _token_importance(key)
    else:
        importance = scores.to(device=device, dtype=torch.float32)
        if importance.ndim != 1 or importance.shape[0] != seq_len:
            raise ValueError(
                f"scores must be 1D of length seq_len={seq_len}, got {tuple(importance.shape)}"
            )

    num_sinks = max(0, min(preserve_sinks, prefix_len, seq_len))
    target_tail = prefix_len - num_sinks

    if seq_len <= prefix_len:
        indices = torch.arange(seq_len, device=device, dtype=torch.long)
    else:
        sink_idx = torch.arange(num_sinks, device=device, dtype=torch.long)
        if target_tail > 0:
            tail_scores = importance[num_sinks:]
            k_take = min(target_tail, tail_scores.shape[0])
            _, top = torch.topk(tail_scores, k_take, dim=-1)
            top = torch.sort(top).values + num_sinks
            indices = torch.cat([sink_idx, top])
        else:
            indices = sink_idx

    idx_expanded = indices.reshape(*([1] * len(leading_dims)), -1, 1).expand(
        *leading_dims, indices.shape[0], head_dim
    )
    k_sel = torch.gather(key, -2, idx_expanded)
    v_sel = torch.gather(value, -2, idx_expanded)

    current_len = indices.shape[0]
    if current_len < prefix_len and pad_to_prefix_len:
        pad_size = prefix_len - current_len
        pad_shape = (*leading_dims, pad_size, head_dim)
        k_sel = torch.cat([k_sel, torch.zeros(pad_shape, dtype=key.dtype, device=device)], dim=-2)
        v_sel = torch.cat([v_sel, torch.zeros(pad_shape, dtype=value.dtype, device=device)], dim=-2)
        pad_idx = torch.arange(current_len, prefix_len, device=device, dtype=torch.long)
        indices = torch.cat([indices, pad_idx])

    k_sel = k_sel.contiguous()
    v_sel = v_sel.contiguous()
    if return_indices:
        return k_sel, v_sel, indices
    return k_sel, v_sel


def spectral_compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_len: int = 32,
    method: CompressionMethod = "select",
    preserve_sinks: int = 1,
    pad_to_prefix_len: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress context KV into a virtual prefix by *token selection* (RoPE-safe).

    Historically this merged keys via truncated SVD. That is incorrect for post-RoPE
    cache keys (phase collapse, C6), so keys are now always **selected**, never
    merged, and values are selected at the same indices to preserve K-V alignment.
    The `method` argument is accepted for backward compatibility and maps to
    leverage-importance selection; pass explicit `scores` to `select_compress_kv`
    for attention-based or extractor-driven importance.

    Returns (k_prefix, v_prefix) of shape (..., prefix_len, head_dim).
    """
    valid = {"select", "svd", "spectral", "joint_svd", "energy", "nystrom"}
    if method.lower() not in valid:
        raise ValueError(
            f"unknown compression method {method!r}; expected one of {sorted(valid)}"
        )
    return select_compress_kv(
        key,
        value,
        prefix_len=prefix_len,
        preserve_sinks=preserve_sinks,
        pad_to_prefix_len=pad_to_prefix_len,
    )


class EngramPrefixTuner:
    """RoPE-correct virtual KV-cache prefix tuner (residual-KV tier).

    Captures the full context KV-cache, selects the most important exact tokens,
    and re-anchors their RoPE phases to contiguous prefix positions so the result
    is a drop-in short prefix for HuggingFace / vLLM / SGLang.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        prefix_len: int = 32,
        method: CompressionMethod = "select",
        preserve_sinks: int = 1,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        rope_theta: float | None = None,
        partial_rotary_factor: float | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.method = method
        self.preserve_sinks = preserve_sinks
        self.device = device
        self.dtype = dtype
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

    def _resolve_rope_config(self, model: nn.Module) -> tuple[float, float] | None:
        if self.rope_theta is not None:
            factor = self.partial_rotary_factor if self.partial_rotary_factor is not None else 1.0
            return float(self.rope_theta), float(factor)
        return rope_config_from_model(model)

    def capture_kv_cache(
        self,
        context_text: str | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        model: nn.Module | None = None,
        **forward_kwargs: Any,
    ) -> DynamicCache:
        """Run a forward pass on context to capture the complete KV-cache.

        Parameters
        ----------
        context_text : str | None, default None
            Raw context text string. Tokenized with self.tokenizer if provided.
        input_ids : torch.Tensor | None, default None
            Token ID tensor of shape (batch_size, seq_len).
        attention_mask : torch.Tensor | None, default None
            Optional attention mask.
        model : nn.Module | None, default None
            Model instance to use (defaults to self.model).

        Returns
        -------
        DynamicCache
            HuggingFace DynamicCache populated with context keys and values.
        """
        active_model = model if model is not None else self.model
        if active_model is None:
            raise ValueError("model must be provided either in __init__ or capture_kv_cache")

        device = self.device
        if device is None:
            try:
                device = next(active_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        encoded: dict[str, Any] = {}
        if context_text is not None:
            if self.tokenizer is None:
                raise ValueError("tokenizer must be provided to process context_text")
            tokens = self.tokenizer(context_text, return_tensors="pt")
            encoded = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in tokens.items()
            }
        elif input_ids is not None:
            encoded["input_ids"] = input_ids.to(device)
            if attention_mask is not None:
                encoded["attention_mask"] = attention_mask.to(device)
        else:
            raise ValueError("either context_text or input_ids must be provided")

        was_training = active_model.training
        active_model.eval()
        try:
            with torch.no_grad():
                outputs = active_model(**encoded, use_cache=True, **forward_kwargs)
                pkv = outputs.past_key_values
                if pkv is None:
                    raise RuntimeError("model forward did not return past_key_values")
                if isinstance(pkv, DynamicCache):
                    return pkv
                # Legacy tuple of tuples format -> DynamicCache
                return DynamicCache.from_legacy_cache(pkv)
        finally:
            active_model.train(was_training)

    def compress_kv_cache(
        self,
        cache: DynamicCache | tuple | torch.Tensor,
        prefix_len: int | None = None,
        method: str | None = None,
        preserve_sinks: int | None = None,
        rope_config: tuple[float, float] | None = None,
        positions: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
    ) -> DynamicCache:
        """Select exact tokens and re-anchor RoPE to contiguous prefix positions.

        Parameters
        ----------
        cache : DynamicCache | tuple | torch.Tensor
            Captured post-RoPE KV-cache (or 6D prefix tensor).
        rope_config : (rope_theta, partial_rotary_factor) | None
            When given, keys are inverted at `positions` and re-applied at 0..B-1.
            When None, keys are selected verbatim (assumes pre-RoPE / position-free).
        positions : torch.Tensor | None
            Original per-token positions of the captured cache (length T). Defaults
            to arange(T) when rope_config is given.
        scores : torch.Tensor | None
            Per-token importance. Shape (T,) is shared across layers; shape
            (num_layers, T) selects each layer with its own scores (SnapKV).
            When None, falls back to key leverage ||k||^2.
        """
        target_p = prefix_len if prefix_len is not None else self.prefix_len
        target_sinks = preserve_sinks if preserve_sinks is not None else self.preserve_sinks

        if isinstance(cache, torch.Tensor) and cache.ndim == 6:
            cache = self.from_tensor(cache)
        if isinstance(cache, (tuple, list)):
            cache = DynamicCache.from_legacy_cache(cache)

        num_layers = len(cache)
        if scores is not None and scores.ndim == 2 and scores.shape[0] != num_layers:
            raise ValueError(
                f"per-layer scores has {scores.shape[0]} rows but cache has {num_layers} layers"
            )
        compressed_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            k, v = cache[layer_idx]
            layer_scores = None
            if scores is not None:
                layer_scores = scores[layer_idx] if scores.ndim == 2 else scores
            k_sel, v_sel, indices = select_compress_kv(
                key=k,
                value=v,
                prefix_len=target_p,
                preserve_sinks=target_sinks,
                scores=layer_scores,
                return_indices=True,
            )
            if rope_config is not None:
                base, factor = rope_config
                seq_len = k.shape[-2]
                if positions is None:
                    orig_pos = torch.arange(seq_len, device=k.device, dtype=torch.long)
                else:
                    orig_pos = positions.to(device=k.device, dtype=torch.long)
                sel_orig_pos = orig_pos[indices]
                new_pos = torch.arange(k_sel.shape[-2], device=k.device, dtype=torch.long)
                k_pre = rope_inverse_kv(k_sel, sel_orig_pos, base, factor)
                k_sel = rope_apply_kv(k_pre, new_pos, base, factor)
            compressed_layers.append((k_sel, v_sel))

        return DynamicCache.from_legacy_cache(tuple(compressed_layers))

    def compile_prefix(
        self,
        context_text: str | None = None,
        input_ids: torch.Tensor | None = None,
        prefix_len: int | None = None,
        method: str | None = None,
        preserve_sinks: int | None = None,
        selection: Literal["leverage", "snapkv"] = "leverage",
        obs_window: int = 32,
        query_text: str | None = None,
        query_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DynamicCache:
        """End-to-end capture + RoPE-correct selection into a resident prefix cache.

        selection="leverage" scores tokens by ||k||^2 (fast, query-agnostic, but
        misses low-norm fact tokens). selection="snapkv" scores tokens by the
        attention they receive from an observation window; it requires eager
        attention. For verbatim fact recall, pass ``query_text`` (or ``query_ids``)
        so the window is the *question* (query-aware SnapKV). Without a query the
        window is the context tail, which is lossy and may drop buried needles.
        """
        active_model = kwargs.get("model", self.model)
        attention_mask = kwargs.get("attention_mask")

        if context_text is not None:
            if self.tokenizer is None:
                raise ValueError("tokenizer must be provided to process context_text")
            device = self.device
            if device is None and active_model is not None:
                try:
                    device = next(active_model.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
            tokens = self.tokenizer(context_text, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device) if device is not None else tokens["input_ids"]
            attention_mask = tokens.get("attention_mask")
            if attention_mask is not None and device is not None:
                attention_mask = attention_mask.to(device)
            context_text = None

        if query_text is not None:
            if self.tokenizer is None:
                raise ValueError("tokenizer must be provided to process query_text")
            if input_ids is None:
                raise ValueError("query_text requires context_text or input_ids")
            q_tokens = self.tokenizer(query_text, return_tensors="pt", add_special_tokens=False)
            query_ids = q_tokens["input_ids"].to(input_ids.device)

        scores = None
        if selection == "snapkv":
            if active_model is None:
                raise ValueError("snapkv selection requires a model")
            if input_ids is None:
                raise ValueError("snapkv selection requires input_ids or context_text")
            scores = snapkv_importance(
                active_model,
                input_ids,
                attention_mask=attention_mask,
                obs_window=obs_window,
                query_ids=query_ids,
            )
        elif selection != "leverage":
            raise ValueError(f"unknown selection {selection!r}; expected 'leverage' or 'snapkv'")

        full_cache = self.capture_kv_cache(
            context_text=context_text,
            input_ids=input_ids,
            **kwargs,
        )
        rope_config = self._resolve_rope_config(active_model) if active_model is not None else None
        return self.compress_kv_cache(
            full_cache,
            prefix_len=prefix_len,
            method=method,
            preserve_sinks=preserve_sinks,
            rope_config=rope_config,
            scores=scores,
        )

    @staticmethod
    def to_tensor(cache: DynamicCache | tuple) -> torch.Tensor:
        """Convert a KV-cache into a single 6D analytical prefix tensor.

        Shape: (num_layers, 2, batch_size, num_kv_heads, prefix_len, head_dim)
        where dimension 1 index 0 corresponds to Key, and index 1 to Value.
        """
        if isinstance(cache, (tuple, list)):
            legacy = cache
        elif isinstance(cache, DynamicCache):
            legacy = tuple(cache[idx] for idx in range(len(cache)))
        else:
            raise TypeError(f"expected DynamicCache or tuple, got {type(cache).__name__}")

        layers_tensors = [
            torch.stack([layer[0], layer[1]], dim=0) for layer in legacy
        ]
        return torch.stack(layers_tensors, dim=0)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> DynamicCache:
        """Convert a 6D tensor into a HuggingFace DynamicCache.

        Expected tensor shape: (num_layers, 2, batch_size, num_kv_heads, prefix_len, head_dim).
        """
        if tensor.ndim != 6:
            raise ValueError(
                f"tensor must be 6D (layers, 2, B, H, P, D), got shape {tensor.shape}"
            )
        if tensor.shape[1] != 2:
            raise ValueError(
                f"dimension 1 must be size 2 (0=Key, 1=Value), got {tensor.shape[1]}"
            )

        num_layers = tensor.shape[0]
        legacy_layers = tuple(
            (tensor[idx, 0].contiguous(), tensor[idx, 1].contiguous())
            for idx in range(num_layers)
        )
        return DynamicCache.from_legacy_cache(legacy_layers)

    def save_prefix(
        self,
        prefix: DynamicCache | tuple | torch.Tensor,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize virtual prefix cache to disk.

        Supports `.safetensors` and PyTorch `.pt` formats.
        """
        dest = Path(path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        prefix_tensor = (
            prefix if (isinstance(prefix, torch.Tensor) and prefix.ndim == 6)
            else self.to_tensor(prefix)
        )
        prefix_tensor = prefix_tensor.detach().contiguous().cpu()

        meta = {
            "num_layers": prefix_tensor.shape[0],
            "batch_size": prefix_tensor.shape[2],
            "num_kv_heads": prefix_tensor.shape[3],
            "prefix_len": prefix_tensor.shape[4],
            "head_dim": prefix_tensor.shape[5],
            "method": self.method,
            **(metadata or {}),
        }

        if dest.suffix == ".safetensors":
            import safetensors.torch

            tensors_dict: dict[str, torch.Tensor] = {}
            for l in range(prefix_tensor.shape[0]):
                tensors_dict[f"layers.{l}.key"] = prefix_tensor[l, 0].contiguous()
                tensors_dict[f"layers.{l}.value"] = prefix_tensor[l, 1].contiguous()

            safetensors_metadata = {k: str(v) for k, v in meta.items()}
            safetensors.torch.save_file(
                tensors_dict,
                str(dest),
                metadata=safetensors_metadata,
            )
        else:
            torch.save(
                {
                    "prefix_tensor": prefix_tensor,
                    "metadata": meta,
                },
                dest,
            )

        return dest

    def load_prefix(
        self,
        path: str | Path,
        device: str | torch.device | None = None,
        return_cache: bool = True,
    ) -> DynamicCache | torch.Tensor:
        """Load virtual prefix cache from disk into DynamicCache or 6D tensor."""
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"prefix file not found: {src}")

        target_device = torch.device(device) if device is not None else torch.device("cpu")

        if src.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(src), framework="pt", device="cpu") as f:
                keys = f.keys()
                if "prefix_tensor" in keys:
                    tensor = f.get_tensor("prefix_tensor").to(target_device)
                else:
                    l = 0
                    layer_pairs: list[torch.Tensor] = []
                    while f"layers.{l}.key" in keys:
                        k_layer = f.get_tensor(f"layers.{l}.key").to(target_device)
                        v_layer = f.get_tensor(f"layers.{l}.value").to(target_device)
                        layer_pairs.append(torch.stack([k_layer, v_layer], dim=0))
                        l += 1
                    if not layer_pairs:
                        raise ValueError(f"no prefix keys found in safetensors file: {src}")
                    tensor = torch.stack(layer_pairs, dim=0)
        else:
            data = torch.load(src, map_location=target_device, weights_only=True)
            tensor = data["prefix_tensor"] if isinstance(data, dict) and "prefix_tensor" in data else data

        if return_cache:
            return self.from_tensor(tensor)
        return tensor

    @staticmethod
    def prepare_generate_inputs(
        input_ids: torch.Tensor,
        prefix_cache: DynamicCache | tuple,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Prepare model inputs for HuggingFace `generate(..., past_key_values=...)`.

        The resident prefix occupies positions 0..B-1 (keys re-RoPE'd at compile
        time), so new tokens continue at cache_position = arange(B, B+q). This is
        phase-consistent — the fix for C6/M6 (no RoPE phase collapse).
        """
        if isinstance(prefix_cache, (tuple, list)):
            past_len = prefix_cache[0][0].shape[-2]
            cache = DynamicCache.from_legacy_cache(prefix_cache)
        elif isinstance(prefix_cache, DynamicCache):
            past_len = prefix_cache.get_seq_length()
            cache = prefix_cache
        else:
            raise TypeError(f"unsupported cache type {type(prefix_cache).__name__}")

        batch_size, query_len = input_ids.shape[:2]
        device = input_ids.device

        cache_position = torch.arange(
            past_len,
            past_len + query_len,
            dtype=torch.long,
            device=device,
        )

        total_len = past_len + query_len
        if attention_mask is None:
            full_mask = torch.ones((batch_size, total_len), dtype=torch.long, device=device)
        else:
            if attention_mask.shape[-1] == query_len:
                prefix_mask = torch.ones((batch_size, past_len), dtype=attention_mask.dtype, device=device)
                full_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            elif attention_mask.shape[-1] == total_len:
                full_mask = attention_mask
            else:
                raise ValueError(
                    f"attention_mask length ({attention_mask.shape[-1]}) must match "
                    f"either query_len ({query_len}) or total_len ({total_len})"
                )

        return {
            "input_ids": input_ids,
            "past_key_values": cache,
            "cache_position": cache_position,
            "attention_mask": full_mask,
        }

    def generate_with_prefix(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        prefix_cache: DynamicCache | tuple,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Run HuggingFace `model.generate` using the virtual prefix cache seamlessly."""
        inputs = self.prepare_generate_inputs(
            input_ids=input_ids,
            prefix_cache=prefix_cache,
            attention_mask=attention_mask,
        )
        return model.generate(**inputs, **generate_kwargs)
