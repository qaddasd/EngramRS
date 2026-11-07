"""Forward-hook activation capture for HuggingFace-style decoder models (legacy module)."""

from __future__ import annotations

from .catchers import (
    AttentionCatcher,
    EngramContextCatcher,
    MLPCatcher,
    capture_tensor,
    find_decoder_layers,
)

__all__ = [
    "AttentionCatcher",
    "EngramContextCatcher",
    "MLPCatcher",
    "capture_tensor",
    "find_decoder_layers",
]
