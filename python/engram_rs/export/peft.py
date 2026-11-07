"""PEFT adapter injection and integration utilities for EngramRS."""

from __future__ import annotations

from pathlib import Path
from torch import nn


def inject_engram(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    """Inject a compiled Engram LoRA adapter into a PyTorch model via PEFT.

    Args:
        model: Base transformer model to adapt.
        adapter_dir: Path to directory containing adapter_model.safetensors
            and adapter_config.json.

    Returns:
        Adapted PeftModel instance.
    """
    from peft import PeftModel

    adapter_path = Path(adapter_dir).expanduser().resolve()
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {adapter_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    return PeftModel.from_pretrained(model, str(adapter_path))
