"""Analytic Virtual KV-Cache Prefix-Tuning for EngramRS.

Directly compiles context key and value tensors (K_C, V_C) into virtual prefix
tokens (Prefix-Tuning) through analytical spectral compression / Nyström approximation,
preserving 100% of the attention softmax non-linearity by directly pre-populating
the KV-cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence
import torch
from torch import nn
from transformers.cache_utils import DynamicCache


def spectral_compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_len: int = 32,
    method: Literal["svd", "spectral", "joint_svd", "energy", "nystrom"] = "svd",
    preserve_sinks: int = 1,
    pad_to_prefix_len: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress context Key and Value tensors along the sequence dimension into virtual prefix tokens.

    Parameters
    ----------
    key : torch.Tensor
        Key tensor of shape (..., seq_len, head_dim), typically (batch_size, num_kv_heads, seq_len, head_dim).
    value : torch.Tensor
        Value tensor of shape matching `key`.
    prefix_len : int, default 32
        Target number of virtual prefix tokens (P).
    method : str, default "svd"
        Compression algorithm:
        - "svd" / "spectral": Truncated SVD along sequence dimension, optimal rank-P
          approximation of K^T K and K^T V.
        - "joint_svd": SVD on concatenated [K, V] sequence subspace.
        - "energy": Chronological energy-weighted pooling across P segments.
        - "nystrom": Landmark sampling based on token leverage scores + dual value projection.
    preserve_sinks : int, default 1
        Number of initial sink tokens (e.g. BOS / position 0) to preserve verbatim
        before compressing the tail context.
    pad_to_prefix_len : bool, default True
        If True, pad compressed sequences shorter than prefix_len with zeros.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (k_prefix, v_prefix) tensors of shape (..., prefix_len, head_dim).
    """
    if key.shape != value.shape:
        raise ValueError(f"key shape {key.shape} does not match value shape {value.shape}")
    if key.ndim < 2:
        raise ValueError(f"key and value must have at least 2 dimensions, got shape {key.shape}")
    if prefix_len <= 0:
        raise ValueError(f"prefix_len must be positive, got {prefix_len}")

    seq_len = key.shape[-2]
    head_dim = key.shape[-1]
    leading_dims = key.shape[:-2]
    orig_dtype = key.dtype
    needs_upcast = orig_dtype in (torch.float16, torch.bfloat16)
    if needs_upcast:
        key = key.float()
        value = value.float()

    # If context is already <= prefix_len and we want to preserve all tokens:
    if seq_len <= prefix_len:
        if seq_len == prefix_len or not pad_to_prefix_len:
            return key.clone().to(orig_dtype), value.clone().to(orig_dtype)
        pad_size = prefix_len - seq_len
        pad_shape = (*leading_dims, pad_size, head_dim)
        k_pad = torch.zeros(pad_shape, dtype=key.dtype, device=key.device)
        v_pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
        return torch.cat([key, k_pad], dim=-2).to(orig_dtype), torch.cat([value, v_pad], dim=-2).to(orig_dtype)

    # Attention sink handling: preserve first S tokens verbatim
    num_sinks = max(0, min(preserve_sinks, prefix_len - 1, seq_len))
    if num_sinks > 0:
        k_sinks = key[..., :num_sinks, :]
        v_sinks = value[..., :num_sinks, :]
        k_rem = key[..., num_sinks:, :]
        v_rem = value[..., num_sinks:, :]
        target_p = prefix_len - num_sinks
    else:
        k_sinks = None
        v_sinks = None
        k_rem = key
        v_rem = value
        target_p = prefix_len

    rem_len = k_rem.shape[-2]
    method_normalized = method.lower()

    if method_normalized in ("svd", "spectral"):
        # Flatten leading dims to (B * H, T, D) for batched SVD
        flat_k = k_rem.reshape(-1, rem_len, head_dim)
        flat_v = v_rem.reshape(-1, rem_len, head_dim)

        # Truncated SVD along sequence dimension: K = U S V^T
        # Left singular vectors U (T, k) span the sequence subspace of maximum variance.
        u, s, vh = torch.linalg.svd(flat_k, full_matrices=False)
        k_comp = min(target_p, s.shape[-1], rem_len)

        # Virtual prefix keys: K_p = S V^T  (shape: B*H, k_comp, D)
        k_p_flat = s[..., :k_comp, None] * vh[..., :k_comp, :]
        # Virtual prefix values: V_p = U^T V (shape: B*H, k_comp, D)
        v_p_flat = torch.matmul(u[..., :k_comp].transpose(-2, -1), flat_v)

        k_p = k_p_flat.reshape(*leading_dims, k_comp, head_dim)
        v_p = v_p_flat.reshape(*leading_dims, k_comp, head_dim)

    elif method_normalized == "joint_svd":
        flat_k = k_rem.reshape(-1, rem_len, head_dim)
        flat_v = v_rem.reshape(-1, rem_len, head_dim)
        joint = torch.cat([flat_k, flat_v], dim=-1)  # (B*H, T, 2D)

        u, s, _ = torch.linalg.svd(joint, full_matrices=False)
        k_comp = min(target_p, s.shape[-1], rem_len)
        u_k = u[..., :k_comp]  # (B*H, T, k_comp)

        k_p_flat = torch.matmul(u_k.transpose(-2, -1), flat_k)
        v_p_flat = torch.matmul(u_k.transpose(-2, -1), flat_v)

        k_p = k_p_flat.reshape(*leading_dims, k_comp, head_dim)
        v_p = v_p_flat.reshape(*leading_dims, k_comp, head_dim)

    elif method_normalized in ("energy", "energy_weighted"):
        # Segment the sequence into target_p chronological bins
        flat_k = k_rem.reshape(-1, rem_len, head_dim)
        flat_v = v_rem.reshape(-1, rem_len, head_dim)
        batch_heads = flat_k.shape[0]

        # Weights based on L2 norm of key vectors
        weights = torch.norm(flat_k, dim=-1, keepdim=True) + 1e-6
        bin_size = rem_len / target_p

        k_bins: list[torch.Tensor] = []
        v_bins: list[torch.Tensor] = []
        for i in range(target_p):
            start = int(round(i * bin_size))
            end = int(round((i + 1) * bin_size))
            end = max(start + 1, min(end, rem_len))

            w_bin = weights[:, start:end, :]  # (B*H, t, 1)
            w_sum = w_bin.sum(dim=-2, keepdim=True) + 1e-8
            w_norm = w_bin / w_sum

            k_bin = (flat_k[:, start:end, :] * w_norm).sum(dim=-2, keepdim=True)
            v_bin = (flat_v[:, start:end, :] * w_norm).sum(dim=-2, keepdim=True)
            k_bins.append(k_bin)
            v_bins.append(v_bin)

        k_p = torch.cat(k_bins, dim=-2).reshape(*leading_dims, target_p, head_dim)
        v_p = torch.cat(v_bins, dim=-2).reshape(*leading_dims, target_p, head_dim)

    elif method_normalized == "nystrom":
        flat_k = k_rem.reshape(-1, rem_len, head_dim)
        flat_v = v_rem.reshape(-1, rem_len, head_dim)
        batch_heads = flat_k.shape[0]

        # Token leverage scores: ||K_t||_2^2
        leverage = torch.sum(flat_k * flat_k, dim=-1)  # (B*H, rem_len)
        k_comp = min(target_p, rem_len)
        _, top_indices = torch.topk(leverage, k_comp, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
        c_k = torch.gather(flat_k, 1, idx_expanded)  # (B*H, k_comp, D)

        # Optimal dual value projection: V_p = (C_K C_K^T + lambda I)^-1 C_K K^T V
        # such that C_K^T V_p ~= K^T V (preserving query-conditioned expectation)
        gram_landmarks = torch.matmul(c_k, c_k.transpose(-2, -1))
        eye = torch.eye(k_comp, device=flat_k.device, dtype=flat_k.dtype).unsqueeze(0)
        gram_reg = gram_landmarks + 1e-3 * eye

        rhs = torch.matmul(c_k, torch.matmul(flat_k.transpose(-2, -1), flat_v))
        try:
            chol = torch.linalg.cholesky(gram_reg)
            v_p_flat = torch.cholesky_solve(rhs, chol)
        except RuntimeError:
            v_p_flat = torch.linalg.solve(gram_reg + 1e-4 * eye, rhs)

        k_p = c_k.reshape(*leading_dims, k_comp, head_dim)
        v_p = v_p_flat.reshape(*leading_dims, k_comp, head_dim)

    else:
        raise ValueError(
            f"unknown compression method {method!r}; expected one of "
            "['svd', 'spectral', 'joint_svd', 'energy', 'nystrom']"
        )

    # Pad compressed tail if shorter than target_p
    current_tail_len = k_p.shape[-2]
    if current_tail_len < target_p and pad_to_prefix_len:
        pad_size = target_p - current_tail_len
        pad_shape = (*leading_dims, pad_size, head_dim)
        k_pad = torch.zeros(pad_shape, dtype=k_p.dtype, device=k_p.device)
        v_pad = torch.zeros(pad_shape, dtype=v_p.dtype, device=v_p.device)
        k_p = torch.cat([k_p, k_pad], dim=-2)
        v_p = torch.cat([v_p, v_pad], dim=-2)

    # Reassemble with preserved attention sinks
    if k_sinks is not None and v_sinks is not None:
        final_k = torch.cat([k_sinks, k_p], dim=-2)
        final_v = torch.cat([v_sinks, v_p], dim=-2)
    else:
        final_k = k_p
        final_v = v_p

    if needs_upcast:
        final_k = final_k.to(orig_dtype)
        final_v = final_v.to(orig_dtype)

    return final_k.contiguous(), final_v.contiguous()


class EngramPrefixTuner:
    """Virtual KV-cache prefix-tuner for LLMs via analytical spectral compression.

    Preserves 100% of attention softmax non-linearity by compiling context activations
    directly into virtual prefix tokens and pre-populating HuggingFace DynamicCache.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        prefix_len: int = 32,
        method: Literal["svd", "spectral", "joint_svd", "energy", "nystrom"] = "svd",
        preserve_sinks: int = 1,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.method = method
        self.preserve_sinks = preserve_sinks
        self.device = device
        self.dtype = dtype

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
    ) -> DynamicCache:
        """Compress full context KV-cache into virtual prefix cache.

        Parameters
        ----------
        cache : DynamicCache | tuple | torch.Tensor
            Captured KV-cache or 6D prefix tensor.
        prefix_len : int | None, default None
            Target prefix length (defaults to self.prefix_len).
        method : str | None, default None
            Compression algorithm (defaults to self.method).
        preserve_sinks : int | None, default None
            Number of sink tokens to keep verbatim (defaults to self.preserve_sinks).

        Returns
        -------
        DynamicCache
            DynamicCache containing the compressed virtual prefix tokens.
        """
        target_p = prefix_len if prefix_len is not None else self.prefix_len
        target_method = method if method is not None else self.method
        target_sinks = preserve_sinks if preserve_sinks is not None else self.preserve_sinks

        # If cache is a 6D tensor of shape (layers, 2, B, H, T, D)
        if isinstance(cache, torch.Tensor) and cache.ndim == 6:
            cache = self.from_tensor(cache)

        if isinstance(cache, (tuple, list)):
            cache = DynamicCache.from_legacy_cache(cache)

        num_layers = len(cache)
        compressed_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(num_layers):
            k, v = cache[layer_idx]
            k_comp, v_comp = spectral_compress_kv(
                key=k,
                value=v,
                prefix_len=target_p,
                method=target_method,  # type: ignore
                preserve_sinks=target_sinks,
            )
            compressed_layers.append((k_comp, v_comp))

        return DynamicCache.from_legacy_cache(tuple(compressed_layers))

    def compile_prefix(
        self,
        context_text: str | None = None,
        input_ids: torch.Tensor | None = None,
        prefix_len: int | None = None,
        method: str | None = None,
        preserve_sinks: int | None = None,
        **kwargs: Any,
    ) -> DynamicCache:
        """End-to-end capture and compilation of context into virtual prefix cache."""
        full_cache = self.capture_kv_cache(
            context_text=context_text,
            input_ids=input_ids,
            **kwargs,
        )
        return self.compress_kv_cache(
            full_cache,
            prefix_len=prefix_len,
            method=method,
            preserve_sinks=preserve_sinks,
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
                tensors_dict[f"layers.{l}.key"] = prefix_tensor[l, 0].contiguous().cpu()
                tensors_dict[f"layers.{l}.value"] = prefix_tensor[l, 1].contiguous().cpu()

            safetensors_metadata = {k: str(v) for k, v in meta.items()}
            safetensors.torch.save_file(
                tensors_dict,
                str(dest),
                metadata=safetensors_metadata,
            )
        else:
            torch.save(
                {
                    "prefix_tensor": prefix_tensor.contiguous(),
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
            data = torch.load(src, map_location=target_device)
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

        Configures proper `cache_position` and `attention_mask` spanning the prefix
        length and new input tokens, avoiding HuggingFace index errors.
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
