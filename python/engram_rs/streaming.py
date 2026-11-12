"""Streaming Covariance Accumulation for arbitrarily long contexts in EngramRS.

Direction 2 in EngramRS:
For linear ridge regression, keeping all N tokens in memory is unnecessary.
Instead, we accumulate sufficient statistics chunk-by-chunk:
    C = sum_{b=1}^B X_b^T X_b,   P = sum_{b=1}^B X_b^T Y_b

Memory footprint is strictly O(d^2) per layer rather than O(N * d), enabling
compilation of 100k+ tokens on standard 4-8 GB VRAM hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence, Union
import numpy as np
import torch
from torch import nn

from .null_space import project_to_null_space
from .randomized_svd import randomized_svd

TensorLike = Union[torch.Tensor, np.ndarray]


class StreamingCovarianceAccumulator:
    """Iteratively accumulate covariance and cross-covariance matrices.

    For each layer l:
        C_l in R^{d x d}: X^T X = sum_b X_b^T X_b
        P_l in R^{d x d}: X^T Y = sum_b X_b^T Y_b

    Memory is strictly bounded by 2 * d^2 * sizeof(float32) per layer,
    independent of total context token count N.
    """

    def __init__(
        self,
        d_model: int | None = None,
        layer_indices: Sequence[int] | None = None,
        use_double_precision: bool = True,
    ) -> None:
        """Initialize the accumulator.

        Args:
            d_model: Feature dimension if known up front.
            layer_indices: Optional list of layer indices to track.
            use_double_precision: If True, accumulates in float64 to prevent
                floating-point drift over 100k+ tokens, returning float32 results.
        """
        self.d_model = d_model
        self.use_double_precision = use_double_precision
        self.layer_indices = list(layer_indices) if layer_indices is not None else None

        self._c: dict[int, np.ndarray] = {}
        self._p: dict[int, np.ndarray] = {}
        self._tokens: dict[int, int] = {}
        self._d_models: dict[int, int] = {}

        if d_model is not None and layer_indices is not None:
            dtype = np.float64 if use_double_precision else np.float32
            for idx in layer_indices:
                self._c[idx] = np.zeros((d_model, d_model), dtype=dtype)
                self._p[idx] = np.zeros((d_model, d_model), dtype=dtype)
                self._tokens[idx] = 0
                self._d_models[idx] = d_model

    @property
    def total_tokens(self) -> int:
        """Maximum tokens accumulated across any layer."""
        return max(self._tokens.values()) if self._tokens else 0

    def memory_bytes(self) -> int:
        """Total memory consumed by accumulated C and P statistics."""
        return sum(c.nbytes + p.nbytes for c, p in zip(self._c.values(), self._p.values()))

    def add_chunk(
        self,
        layer_idx: int,
        x_chunk: TensorLike,
        y_chunk: TensorLike,
    ) -> None:
        """Accumulate one chunk of activations X_b and targets Y_b for a layer.

        Args:
            layer_idx: Layer index integer.
            x_chunk: Hidden states matrix of shape [N_b, d].
            y_chunk: Attention outputs matrix of shape [N_b, d].
        """
        if isinstance(x_chunk, torch.Tensor):
            x_np = x_chunk.detach().cpu().to(torch.float32).numpy()
        else:
            x_np = np.asarray(x_chunk, dtype=np.float32)

        if isinstance(y_chunk, torch.Tensor):
            y_np = y_chunk.detach().cpu().to(torch.float32).numpy()
        else:
            y_np = np.asarray(y_chunk, dtype=np.float32)

        if x_np.ndim != 2 or y_np.ndim != 2:
            raise ValueError(
                f"x_chunk and y_chunk must be 2D matrices, got {x_np.shape} and {y_np.shape}"
            )
        if x_np.shape != y_np.shape:
            raise ValueError(
                f"x_chunk shape {x_np.shape} does not match y_chunk shape {y_np.shape}"
            )

        n_b, d = x_np.shape
        if n_b == 0:
            return

        dtype = np.float64 if self.use_double_precision else np.float32
        x_mat = x_np.astype(dtype, copy=False)
        y_mat = y_np.astype(dtype, copy=False)

        c_chunk = x_mat.T @ x_mat
        p_chunk = x_mat.T @ y_mat

        if layer_idx not in self._c:
            self._c[layer_idx] = c_chunk
            self._p[layer_idx] = p_chunk
            self._tokens[layer_idx] = n_b
            self._d_models[layer_idx] = d
        else:
            if self._d_models[layer_idx] != d:
                raise ValueError(
                    f"dimension mismatch for layer {layer_idx}: expected {self._d_models[layer_idx]}, got {d}"
                )
            self._c[layer_idx] += c_chunk
            self._p[layer_idx] += p_chunk
            self._tokens[layer_idx] += n_b

    def accumulate_from_model(
        self,
        model: nn.Module,
        tokenizer: Any,
        text_or_tokens: str | Sequence[int] | torch.Tensor,
        chunk_size: int = 2048,
        overlap_tokens: int = 0,
        device: torch.device | str | None = None,
        layer_indices: Sequence[int] | None = None,
    ) -> int:
        """Stream long context through model forward passes and accumulate C and P.

        Uses a single persistent KV cache across chunks so causal attention and RoPE
        positions are continuous (each token attends to the full preceding context).
        The accumulator statistics stay O(d^2) per layer; the forward KV cache is O(N)
        as required for exact causal attention.

        Args:
            model: HuggingFace decoder model.
            tokenizer: Model tokenizer.
            text_or_tokens: Raw context text string or token IDs list/tensor.
            chunk_size: Chunk size in tokens (default 2048).
            overlap_tokens: Unused (kept for API compatibility); continuous cache
                makes overlap re-feeding unnecessary and harmful.
            device: Model device.
            layer_indices: Optional subset of decoder layers.

        Returns:
            Total number of tokens processed.
        """
        from .hooks import EngramContextCatcher

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        # Obtain token IDs
        if isinstance(text_or_tokens, str):
            encoded = tokenizer(text_or_tokens, return_tensors="pt")
            input_ids = encoded["input_ids"]
        elif isinstance(text_or_tokens, torch.Tensor):
            input_ids = text_or_tokens
        else:
            input_ids = torch.tensor([list(text_or_tokens)], dtype=torch.long)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        total_tokens = input_ids.shape[-1]
        catcher = EngramContextCatcher(model, layer_indices=layer_indices)
        was_training = model.training
        model.eval()

        # Causal continuity: a single persistent KV cache spans all chunks so that
        # every token attends to the full preceding context and RoPE positions are
        # continuous. Independent per-chunk forwards (the old behavior) destroyed
        # cross-chunk attention and produced semantically wrong activations.
        from transformers import DynamicCache

        cache = DynamicCache()
        # overlap_tokens is accepted for API compatibility but unused: with a
        # continuous cache, re-feeding overlap would double-count tokens.
        _ = overlap_tokens

        try:
            for start_idx in range(0, total_tokens, chunk_size):
                end_idx = min(start_idx + chunk_size, total_tokens)
                chunk_ids = input_ids[:, start_idx:end_idx].to(device)
                cache_position = torch.arange(start_idx, end_idx, device=device)
                attention_mask = torch.ones(1, end_idx, dtype=torch.long, device=device)

                with torch.no_grad(), catcher:
                    outputs = model(
                        input_ids=chunk_ids,
                        past_key_values=cache,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    del outputs

                hidden_states, attention_outputs = catcher.get_activations()
                for layer_idx, hidden, attention in zip(
                    catcher.layer_indices, hidden_states, attention_outputs
                ):
                    self.add_chunk(layer_idx, hidden, attention)

                catcher.clear()
                del chunk_ids
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            return total_tokens
        finally:
            catcher.clear()
            del cache
            model.train(was_training)

    def solve(
        self,
        lambda_reg: float = 1e-4,
        rank: int = 16,
        method: str = "randomized_svd",
        n_iter: int = 2,
        strict_null_projection: bool = False,
        null_states: dict[int, TensorLike] | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Compute the closed-form ridge regression adapter for all layers.

        Solves (C_l + lambda_reg * I) * W_l = P_l, then factors W_l into
        lora_A [rank, d_model] and lora_B [d_model, rank].

        Args:
            lambda_reg: Ridge regularization parameter (lambda > 0).
            rank: LoRA rank r.
            method: "randomized_svd" (fast Halko-Martinsson-Tropp) or "svd" (deterministic).
            n_iter: Power iterations for randomized SVD.
            strict_null_projection: If True, projects adapter into null-space of null_states.
            null_states: Dict mapping layer_idx -> background activation matrix X_null.

        Returns:
            dict mapping layer_idx -> (lora_a, lora_b) NumPy arrays.
        """
        if lambda_reg <= 0.0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        compiled_layers: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for layer_idx in sorted(self._c.keys()):
            c_mat = self._c[layer_idx]
            p_mat = self._p[layer_idx]
            d = self._d_models[layer_idx]

            # Relative ridge: lambda scales with the covariance energy so it remains
            # meaningful across activation magnitudes (absolute 1e-4 is negligible
            # for LLM activations with ||x||^2 ~ 1e3-1e6).
            lambda_eff = lambda_reg * float(np.trace(c_mat)) / max(1, d)
            reg_gram = c_mat + lambda_eff * np.eye(d, dtype=c_mat.dtype)
            try:
                # Solve (C + lambda I) W = P
                weights = np.linalg.solve(reg_gram, p_mat).astype(np.float32)
            except np.linalg.LinAlgError:
                # Fallback to least-squares if near singular
                weights, _, _, _ = np.linalg.lstsq(reg_gram, p_mat, rcond=None)
                weights = weights.astype(np.float32)

            # Project once into the null-space of background activations, then
            # factorize. Projecting again after factorization (the old behavior)
            # double-attenuated the update.
            if strict_null_projection and null_states is not None and layer_idx in null_states:
                x_null = null_states[layer_idx]
                weights = project_to_null_space(weights, x_null, side="right")

            # Low-rank factorize W ~ U S V^T into lora_A and lora_B
            # SVD: W = U S V^T
            # In EngramRS / PEFT:
            # lora_A has shape [rank, d], row k is sqrt(s_k) * U[:, k]^T
            # lora_B has shape [d, rank], col k is sqrt(s_k) * V[:, k]
            # so (lora_A)^T @ (lora_B)^T = U S V^T = W.
            if method == "randomized_svd":
                u, s, vh = randomized_svd(weights, rank=rank, n_iter=n_iter)
            else:
                u, s, vh = np.linalg.svd(weights, full_matrices=False)
                u = u[:, :rank]
                s = s[:rank]
                vh = vh[:rank, :]

            effective_rank = min(rank, len(s))
            lora_a = np.zeros((rank, d), dtype=np.float32)
            lora_b = np.zeros((d, rank), dtype=np.float32)

            scales = np.sqrt(np.maximum(s[:effective_rank], 0.0))
            for k in range(effective_rank):
                lora_a[k, :] = scales[k] * u[:, k]
                lora_b[:, k] = scales[k] * vh[k, :]

            compiled_layers[layer_idx] = (lora_a, lora_b)

        return compiled_layers

    def save_adapter(
        self,
        output_dir: str | Path,
        rank: int = 16,
        lambda_reg: float = 1e-4,
        base_model_name_or_path: str = "",
        method: str = "randomized_svd",
        strict_null_projection: bool = False,
        null_states: dict[int, TensorLike] | None = None,
    ) -> Path:
        """Solve and serialize adapter to PEFT safetensors and config.

        Creates adapter_model.safetensors and adapter_config.json.

        Args:
            output_dir: Output directory path.
            rank: Target rank.
            lambda_reg: Ridge regularizer.
            base_model_name_or_path: Name of base model for PEFT config.
            method: SVD factorization method.
            strict_null_projection: Strict null-space projection flag.
            null_states: Optional background representations per layer.

        Returns:
            Path to output directory.
        """
        import struct

        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        layers = self.solve(
            lambda_reg=lambda_reg,
            rank=rank,
            method=method,
            strict_null_projection=strict_null_projection,
            null_states=null_states,
        )

        tensor_dict: dict[str, np.ndarray] = {}
        for layer_idx, (lora_a, lora_b) in layers.items():
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_"
            tensor_dict[f"{prefix}A.weight"] = np.ascontiguousarray(lora_a, dtype=np.float32)
            tensor_dict[f"{prefix}B.weight"] = np.ascontiguousarray(lora_b, dtype=np.float32)

        # Write safetensors
        header: dict[str, Any] = {}
        data_buffers: list[bytes] = []
        offset = 0
        for name, tensor in tensor_dict.items():
            raw = tensor.tobytes()
            nbytes = len(raw)
            header[name] = {
                "dtype": "F32",
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + nbytes],
            }
            data_buffers.append(raw)
            offset += nbytes

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Align header length to 8 bytes if needed
        header_len = len(header_json)
        pad = (8 - (header_len % 8)) % 8
        if pad > 0:
            header_json += b" " * pad
            header_len += pad

        safetensors_path = out_path / "adapter_model.safetensors"
        with open(safetensors_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)
            for buf in data_buffers:
                f.write(buf)

        # Write adapter_config.json in standard PEFT format
        config = {
            "base_model_name_or_path": base_model_name_or_path,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": rank,
            "rank_pattern": {},
            "target_modules": ["o_proj"],
            "task_type": "CAUSAL_LM",
        }
        with open(out_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return out_path
