"""Analytical Zero-Grad Model Merge for Engram / LoRA adapters.

Analytically merges LoRA weight increments (\\Delta W = (lora_alpha / r) * (lora_B @ lora_A))
directly into the base model weights without autograd or backpropagation, saving
a standalone Hugging Face model that requires no PEFT dependency during inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def compute_delta_w(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    r: int,
    fan_in_fan_out: bool = False,
) -> torch.Tensor:
    """Compute the analytical weight update \\Delta W for a LoRA layer.

    In standard LoRA:
        h = W x + (alpha / r) * B A x = (W + (alpha / r) * B A) x
    where:
        lora_A has shape (r, in_features)
        lora_B has shape (out_features, r)
        B @ A has shape (out_features, in_features)
        scaling = alpha / r

    Args:
        lora_a: Down-projection tensor of shape [r, in_features].
        lora_b: Up-projection tensor of shape [out_features, r].
        alpha: LoRA scaling hyperparameter (lora_alpha).
        r: LoRA rank (r).
        fan_in_fan_out: Whether the target module weights are stored transposed (e.g., Conv1D).

    Returns:
        Tensor of shape [out_features, in_features] (or transposed if fan_in_fan_out is True).
    """
    if r <= 0:
        raise ValueError(f"LoRA rank r must be positive, got {r}")
    if alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {alpha}")

    scaling = float(alpha) / float(r)
    # Ensure tensors are float32 for high precision during matmul
    orig_dtype = lora_a.dtype
    a_f32 = lora_a.to(torch.float32)
    b_f32 = lora_b.to(torch.float32)

    delta_w = (b_f32 @ a_f32) * scaling
    if fan_in_fan_out:
        delta_w = delta_w.t()

    return delta_w.to(dtype=orig_dtype)


def normalize_module_name(name: str) -> str:
    """Strip standard PEFT and model wrapping prefixes for clean module matching."""
    prefixes = (
        "base_model.model.",
        "base_model.",
        "model.",
        "transformer.",
    )
    result = name
    for prefix in prefixes:
        if result.startswith(prefix):
            result = result[len(prefix) :]
    return result


def extract_lora_pairs(
    adapter_tensors: dict[str, torch.Tensor]
) -> dict[str, dict[str, torch.Tensor]]:
    """Group adapter tensors into matching (lora_A, lora_B) pairs per module.

    Keys in PEFT safetensors are typically styled as:
        'base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight'
        'base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight'
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter_tensors.items():
        if ".lora_A.weight" in key or ".lora_A." in key:
            base_key = key.replace(".lora_A.weight", "").replace(".lora_A", "")
            pairs.setdefault(base_key, {})["A"] = tensor
        elif ".lora_B.weight" in key or ".lora_B." in key:
            base_key = key.replace(".lora_B.weight", "").replace(".lora_B", "")
            pairs.setdefault(base_key, {})["B"] = tensor

    # Validate each pair has both A and B
    for base_key, components in pairs.items():
        if "A" not in components or "B" not in components:
            missing = "A" if "A" not in components else "B"
            raise ValueError(f"LoRA module '{base_key}' is missing component '{missing}'")

    return pairs


def find_target_parameter(
    model: nn.Module,
    adapter_module_name: str,
) -> tuple[str, nn.Parameter] | None:
    """Find the base model parameter corresponding to an adapter module.

    Matches by normalized suffix or direct submodule path.
    """
    norm_adapter = normalize_module_name(adapter_module_name)
    target_param_name = norm_adapter + ".weight"

    named_params = dict(model.named_parameters())

    # 1. Exact match with .weight
    if target_param_name in named_params:
        return target_param_name, named_params[target_param_name]

    # 2. Suffix match (e.g., model.layers.0.self_attn.o_proj.weight vs layers.0.self_attn.o_proj.weight)
    for name, param in named_params.items():
        if name == target_param_name or name.endswith("." + target_param_name):
            return name, param
        norm_name = normalize_module_name(name)
        if norm_name == target_param_name or norm_name.endswith(target_param_name):
            return name, param

    return None


def merge_lora_into_model(
    model: nn.Module,
    adapter_path: str | Path,
    inplace: bool = True,
) -> nn.Module:
    """Analytically merge LoRA weights directly into a PyTorch model's parameters.

    Operates in zero-grad mode with no computational graph construction.

    Args:
        model: Hugging Face model or nn.Module containing the base parameters.
        adapter_path: Directory containing adapter_model.safetensors and adapter_config.json.
        inplace: Whether to modify model parameters in-place.

    Returns:
        The merged model.
    """
    adapter_dir = Path(adapter_path).expanduser().resolve()
    config_file = adapter_dir / "adapter_config.json"
    weights_file = adapter_dir / "adapter_model.safetensors"

    if not config_file.is_file():
        raise FileNotFoundError(f"Missing adapter configuration file: {config_file}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"Missing adapter weights file: {weights_file}")

    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    r = int(config.get("r", 16))
    alpha = float(config.get("lora_alpha", r))
    fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

    from safetensors.torch import load_file

    adapter_tensors = load_file(str(weights_file))
    pairs = extract_lora_pairs(adapter_tensors)

    if not pairs:
        raise ValueError(f"No valid LoRA (lora_A, lora_B) weight pairs found in {weights_file}")

    merged_count = 0
    with torch.no_grad():
        for adapter_mod_name, pair in pairs.items():
            lora_a = pair["A"]
            lora_b = pair["B"]

            target = find_target_parameter(model, adapter_mod_name)
            if target is None:
                raise KeyError(
                    f"Could not find matching base model parameter for adapter module '{adapter_mod_name}'. "
                    f"Available model parameters: {list(dict(model.named_parameters()).keys())[:5]}..."
                )

            param_name, param = target
            delta_w = compute_delta_w(
                lora_a, lora_b, alpha=alpha, r=r, fan_in_fan_out=fan_in_fan_out
            )

            if delta_w.shape != param.shape:
                # Check if transposed
                if delta_w.t().shape == param.shape:
                    delta_w = delta_w.t()
                else:
                    raise ValueError(
                        f"Shape mismatch merging into parameter '{param_name}': "
                        f"base parameter shape is {param.shape}, but computed \\Delta W shape is {delta_w.shape}"
                    )

            delta_w_cast = delta_w.to(device=param.device, dtype=param.dtype)
            if inplace:
                param.data.add_(delta_w_cast)
            else:
                param.data = param.data + delta_w_cast
            merged_count += 1

    return model


def merge_adapter_to_base(
    base_model_name_or_path: str | Path | nn.Module,
    adapter_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    torch_dtype: Any = None,
    tokenizer_name_or_path: str | Path | None = None,
    safe_serialization: bool = True,
) -> str:
    """Load base model, merge LoRA adapter analytically, and save standalone HF model.

    The saved model has no dependencies on PEFT and can be loaded directly with
    AutoModelForCausalLM.from_pretrained(output_dir).

    Args:
        base_model_name_or_path: Hugging Face model identifier, path, or instantiated model.
        adapter_path: Path to directory containing adapter_model.safetensors and adapter_config.json.
        output_dir: Output directory where the merged standalone model will be saved.
        device: Device on which to load the base model (default 'cpu').
        torch_dtype: Optional torch dtype (e.g. torch.float16, torch.bfloat16).
        tokenizer_name_or_path: Optional path/name for tokenizer (defaults to base_model_name_or_path).
        safe_serialization: Whether to save model weights using safetensors.

    Returns:
        Absolute path to the output directory as a string.
    """
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = None

    if isinstance(base_model_name_or_path, nn.Module):
        model = base_model_name_or_path
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path_str = str(base_model_name_or_path)
        tok_path_str = str(tokenizer_name_or_path or base_model_name_or_path)

        # Load tokenizer if available
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path_str)
        except Exception:
            tokenizer = None

        # Load model
        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if device != "auto":
            model = AutoModelForCausalLM.from_pretrained(model_path_str, **load_kwargs)
            model.to(device)
        else:
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(model_path_str, **load_kwargs)

    # Perform zero-grad merge
    merge_lora_into_model(model, adapter_path=adapter_path, inplace=True)

    # Save standalone merged model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(
            str(output_path),
            safe_serialization=safe_serialization,
        )
    else:
        # Fallback for plain nn.Module
        from safetensors.torch import save_file

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict, str(output_path / "model.safetensors"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_path))

    return str(output_path)


def get_adapter_scaling(cfg: dict[str, Any], actual_r: int, a_key: str) -> float:
    """Retrieve scaling factor alpha / r for a given LoRA module, respecting rank_pattern and alpha_pattern."""
    norm_key = normalize_module_name(a_key.replace(".lora_A.weight", "").replace(".lora_A", ""))

    rank_pattern = cfg.get("rank_pattern", {})
    r = rank_pattern.get(norm_key, rank_pattern.get(a_key, actual_r))
    if not isinstance(r, (int, float)) or r <= 0:
        r = actual_r

    alpha_pattern = cfg.get("alpha_pattern", {})
    alpha = alpha_pattern.get(norm_key, alpha_pattern.get(a_key, cfg.get("lora_alpha", float(r))))
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        alpha = float(r)

    return float(alpha) / float(r)


def combine_lora_adapters(
    adapter_paths: Sequence[str | Path],
    output_dir: str | Path,
    target_rank: int = 16,
    target_alpha: float | None = None,
) -> str:
    """Combine two or more LoRA adapters into a single compact adapter via SVD and orthogonalization.

    Incorporates exact alpha / r and rank_pattern scaling per adapter and normalizes by
    output scaling, guaranteeing zero amplitude degradation when loaded through PEFT.

    Args:
        adapter_paths: List of directory paths to existing LoRA adapters.
        output_dir: Output path where merged adapter will be written.
        target_rank: Maximum LoRA rank for the merged adapter.
        target_alpha: Optional LoRA alpha for the merged adapter (defaults to target_rank).

    Returns:
        Absolute path to the merged adapter directory.
    """
    from safetensors.torch import load_file, save_file
    import numpy as np

    if not adapter_paths:
        raise ValueError("adapter_paths must contain at least one adapter path")

    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    loaded_adapters = []
    configs = []
    for p in adapter_paths:
        p = Path(p).expanduser().resolve()
        cfg_file = p / "adapter_config.json"
        sf_file = p / "adapter_model.safetensors"
        if not cfg_file.is_file() or not sf_file.is_file():
            raise FileNotFoundError(f"adapter in {p} missing adapter_config.json or adapter_model.safetensors")
        with open(cfg_file, "r", encoding="utf-8") as f:
            configs.append(json.load(f))
        loaded_adapters.append(load_file(str(sf_file)))

    if target_alpha is None:
        target_alpha = float(target_rank)

    base_config = dict(configs[0])
    base_config["r"] = target_rank
    base_config["lora_alpha"] = target_alpha
    base_config.pop("rank_pattern", None)
    base_config.pop("alpha_pattern", None)

    merged_tensors: dict[str, torch.Tensor] = {}

    # Group keys by parameter base
    all_keys = set()
    for ad in loaded_adapters:
        all_keys.update(ad.keys())

    a_keys = [k for k in all_keys if ".lora_A.weight" in k]

    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")

        a_list = []
        b_list = []
        for idx, ad in enumerate(loaded_adapters):
            if a_key in ad and b_key in ad:
                a_raw = ad[a_key].to(torch.float32)
                b_raw = ad[b_key].to(torch.float32)
                actual_r = int(a_raw.shape[0])
                scaling_k = get_adapter_scaling(configs[idx], actual_r, a_key)
                mult_k = float(np.sqrt(scaling_k))
                a_list.append(a_raw * mult_k)
                b_list.append(b_raw * mult_k)

        if not a_list:
            continue

        # Concatenate: B_cat = [B1, B2], A_cat = [A1; A2]
        b_cat = torch.cat(b_list, dim=1)  # [d_out, total_r]
        a_cat = torch.cat(a_list, dim=0)  # [total_r, d_in]

        # Thin QR: B_cat = Q_b * R_b, A_cat^T = Q_a * R_a
        q_b, r_b = torch.linalg.qr(b_cat)
        q_a, r_a = torch.linalg.qr(a_cat.t())

        # Core matrix M = R_b * R_a^T of shape [p, p]
        m_core = r_b @ r_a.t()
        u, s, vh = torch.linalg.svd(m_core, full_matrices=False)

        r = min(target_rank, len(s))
        s_sqrt = torch.sqrt(s[:r])

        b_merged = q_b @ (u[:, :r] * s_sqrt)
        a_merged = (vh[:r, :].t() * s_sqrt).t() @ q_a.t()

        # Normalize factors by output scaling factor (target_alpha / target_rank)
        # so that PEFT computation: (target_alpha / target_rank) * (B_merged @ A_merged) == Delta W_total
        target_scaling = float(target_alpha) / float(target_rank)
        out_mult = 1.0 / float(np.sqrt(target_scaling))
        b_merged = b_merged * out_mult
        a_merged = a_merged * out_mult

        merged_tensors[a_key] = a_merged.contiguous()
        merged_tensors[b_key] = b_merged.contiguous()

    save_file(merged_tensors, str(out_path / "adapter_model.safetensors"))
    with open(out_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(base_config, f, indent=2)

    return str(out_path)


def unlearn_concept_from_adapter(
    adapter_path: str | Path,
    concept_activations: np.ndarray,
    output_dir: str | Path,
    tolerance: float = 1e-5,
) -> str:
    """Unlearn/scrub a concept from a LoRA adapter by orthogonal complement projection.

    Guarantees that for any representation x in the concept space:
        x @ Delta W_unlearned^T = 0.0000

    Args:
        adapter_path: Path to existing LoRA adapter.
        concept_activations: [K, d_in] matrix of activations representing the concept.
        output_dir: Output directory for the scrubbed adapter.
        tolerance: Singular value cutoff for the concept basis.

    Returns:
        Path to the output scrubbed adapter directory.
    """
    from safetensors.torch import load_file, save_file
    from .null_space import project_lora_to_null_space

    in_path = Path(adapter_path).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(in_path / "adapter_model.safetensors"))
    a_keys = [k for k in tensors.keys() if ".lora_A.weight" in k]

    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key in tensors:
            if tensors[a_key].shape[1] == concept_activations.shape[1]:
                a_proj, _ = project_lora_to_null_space(
                    tensors[a_key], tensors[b_key], concept_activations, rcond=tolerance
                )
                tensors[a_key] = a_proj

    save_file(tensors, str(out_path / "adapter_model.safetensors"))
    cfg_in = in_path / "adapter_config.json"
    if cfg_in.is_file():
        (out_path / "adapter_config.json").write_bytes(cfg_in.read_bytes())

    return str(out_path)
