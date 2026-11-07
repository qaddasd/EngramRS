"""Causal Layer Sensitivity Profiler for Dynamic LoRA Allocation.

Measures the factual associative sensitivity of each decoder layer:
    S_l = ||Delta Y_l||_F / ||H_l||_F
where Delta Y_l is the contextual attention shift and H_l is the hidden state.

Layers with higher sensitivity S_l carry higher information density and associative
memory capacity, making them prime targets for higher LoRA ranks or selective adaptation.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import nn

from .catchers import EngramContextCatcher


class LayerSensitivityProfiler:
    """Profiles decoder layers by their factual memory capacity and sensitivity."""

    @staticmethod
    def profile(
        model: nn.Module,
        tokenizer: Any,
        text: str,
        device: torch.device | str | None = None,
    ) -> dict[int, float]:
        """Compute the sensitivity score S_l for every decoder layer in the model.

        Args:
            model: HuggingFace causal decoder model.
            tokenizer: Tokenizer.
            text: Context or calibration text to measure sensitivity on.
            device: Device to place tensors on.

        Returns:
            dict mapping layer_index -> sensitivity ratio S_l >= 0.
        """
        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        catcher = EngramContextCatcher(model, targets=("o_proj",))
        was_training = model.training
        model.eval()

        try:
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in encoded.items()
            }
            with torch.no_grad(), catcher:
                outputs = model(**encoded, use_cache=False)
                del outputs

            hidden_states, attention_outputs = catcher.get_activations("o_proj")
            scores: dict[int, float] = {}

            for layer_idx, h, y in zip(catcher.layer_indices, hidden_states, attention_outputs):
                norm_h = float(np.linalg.norm(h, "fro"))
                norm_y = float(np.linalg.norm(y, "fro"))
                scores[layer_idx] = float(norm_y / (norm_h + 1e-12))

            return scores
        finally:
            catcher.clear()
            model.train(was_training)

    @staticmethod
    def top_k_layers(scores: dict[int, float], top_k: int = 12) -> list[int]:
        """Return the indices of the top-K most sensitive layers."""
        sorted_layers = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        return sorted_layers[:top_k]

    @staticmethod
    def adaptive_rank_pattern(
        scores: dict[int, float],
        base_rank: int = 16,
        min_rank: int = 4,
        max_rank: int = 32,
    ) -> dict[str, int]:
        """Compute PEFT rank_pattern dictionary scaling rank with layer sensitivity.

        Layers with above-average sensitivity receive higher rank, while low-sensitivity
        layers receive lower rank, minimizing parameter count while maximizing accuracy.
        """
        values = list(scores.values())
        if not values:
            return {}
        median_val = float(np.median(values))
        if median_val <= 1e-12:
            median_val = 1.0

        pattern: dict[str, int] = {}
        for layer_idx, score in scores.items():
            relative = score / median_val
            allocated = int(np.clip(np.round(base_rank * relative), min_rank, max_rank))
            pattern[f"layers.{layer_idx}.self_attn.o_proj"] = allocated
            pattern[f"layers.{layer_idx}.mlp.down_proj"] = allocated
        return pattern
