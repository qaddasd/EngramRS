from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .catchers.base import find_decoder_layers
from .engram_rs import compile_all_layers_to_safetensors
from .hooks import EngramContextCatcher


def resolve_layer_indices(
    model: nn.Module,
    layer_indices: Iterable[int] | None,
    all_layers: bool = False,
) -> tuple[int, ...]:
    """Select target layers for adaptation.

    Defaults to middle semantic layers [0.25 * L, 0.80 * L] where factual and
    associative memory resides, avoiding perturbation of shallow syntax and
    deep unembedding layers. For small models (<= 4 layers), adapts all layers.
    """
    layers = find_decoder_layers(model)
    total_layers = len(layers)
    if layer_indices is not None:
        return tuple(layer_indices)
    if all_layers or total_layers <= 4:
        return tuple(range(total_layers))
    start = int(np.floor(0.25 * total_layers))
    end = int(np.ceil(0.80 * total_layers))
    return tuple(range(start, end))


class EngramCompiler:
    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.last_compile_time_ms: float | None = None
        self.last_prompt_tokens: int | None = None

    def compile_text(
        self,
        context_text: str,
        output_dir: str,
        rank: int = 16,
        targets: Iterable[str] = ("o_proj", "down_proj"),
        layer_indices: Iterable[int] | None = None,
        all_layers: bool = False,
        contrastive_delta: bool = False,
        surprise_weighting: bool = False,
        scale_depth: bool = True,
        energy_threshold: float | None = None,
        min_rank: int = 1,
        max_rank: int | None = None,
        max_anchors: int = 64,
        lambda_reg: float = 1e-4,
        lambda_null: float = 1.0,
        null_states: Any = None,
        strict_null_projection: bool = False,
    ) -> str:
        if not isinstance(context_text, str) or not context_text.strip():
            raise ValueError("context_text must be a non-empty string")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if energy_threshold is not None:
            if not isinstance(energy_threshold, (int, float)) or energy_threshold <= 0.0 or energy_threshold > 1.0:
                raise ValueError("energy_threshold must be a float in (0.0, 1.0]")
            energy_threshold = float(energy_threshold)
        if max_rank is None:
            max_rank = rank
        if isinstance(min_rank, bool) or not isinstance(min_rank, int) or min_rank <= 0:
            raise ValueError("min_rank must be a positive integer")
        if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < min_rank:
            raise ValueError("max_rank must be an integer >= min_rank")

        destination = Path(output_dir).expanduser().resolve()
        device = next(self.model.parameters()).device

        selected_layers = resolve_layer_indices(self.model, layer_indices, all_layers=all_layers)
        catcher = EngramContextCatcher(self.model, layer_indices=selected_layers, targets=targets)
        was_training = self.model.training
        encoded: Any = None
        token_weights: np.ndarray | None = None

        try:
            self.model.eval()
            encoded = self.tokenizer(context_text, return_tensors="pt")
            if not hasattr(encoded, "items"):
                raise TypeError("tokenizer must return a mapping of model inputs")
            encoded = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
            input_ids = encoded.get("input_ids")
            if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
                raise ValueError("tokenizer returned no input_ids")
            self.last_prompt_tokens = int(input_ids.shape[-1])

            with torch.no_grad(), catcher:
                outputs = self.model(**encoded, use_cache=False)
                if surprise_weighting and hasattr(outputs, "logits"):
                    # Calculate per-token cross entropy loss for causal surprise weighting
                    logits = outputs.logits
                    seq_len = int(logits.shape[1])
                    if seq_len > 1:
                        shift_logits = logits[0, :-1, :].contiguous()
                        shift_labels = input_ids[0, 1:].contiguous()
                        ce_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
                        ce_f32 = ce_loss.detach().to(device="cpu", dtype=torch.float32).numpy()
                        # Causal alignment: state at t predicted token t+1.
                        # ce_f32[t] is the loss of predicting token t+1 from state t.
                        # Apply exponential smoothing: w_t = sqrt(max(eps, 0.7 * L(t+1) + 0.3 * L(t)))
                        eps = 1e-4
                        w = np.zeros(seq_len, dtype=np.float32)
                        w[0] = np.sqrt(max(eps, min(float(ce_f32[0]), 50.0)))
                        for t in range(1, seq_len - 1):
                            smoothed_ce = 0.7 * float(ce_f32[t]) + 0.3 * float(ce_f32[t - 1])
                            w[t] = np.sqrt(max(eps, min(smoothed_ce, 50.0)))
                        # Last token N-1 has no t+1 token, so w[N-1] = w[N-2]
                        w[seq_len - 1] = w[seq_len - 2]
                        token_weights = w
                    else:
                        token_weights = np.ones(seq_len, dtype=np.float32)
                del outputs

            if hasattr(encoded, "clear"):
                encoded.clear()
            encoded = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Retrieve activations, using Cross-Sublayer Contrastive Delta if enabled
            if contrastive_delta and "down_proj" in catcher.targets and "o_proj" in catcher.targets:
                all_activations = catcher.get_contrastive_target_activations()
            else:
                all_activations = catcher.get_all_activations()

            target_inputs = {target: acts[0] for target, acts in all_activations.items()}
            target_outputs = {target: acts[1] for target, acts in all_activations.items()}

            # Apply surprise weighting if computed.
            # Row weight w enters least squares as sqrt(w) on both X and Y,
            # so the effective weight is w (not w²).
            if token_weights is not None:
                token_weights_f32 = np.ascontiguousarray(
                    np.sqrt(token_weights), dtype=np.float32
                )
                for target in target_inputs:
                    target_inputs[target] = [
                        np.ascontiguousarray(mat * token_weights_f32[:, None], dtype=np.float32)
                        if mat.shape[0] == len(token_weights_f32)
                        else mat
                        for mat in target_inputs[target]
                    ]
                    target_outputs[target] = [
                        np.ascontiguousarray(mat * token_weights_f32[:, None], dtype=np.float32)
                        if mat.shape[0] == len(token_weights_f32)
                        else mat
                        for mat in target_outputs[target]
                    ]

            config = getattr(self.model, "config", None)
            base_model_name = str(
                getattr(self.model, "name_or_path", "")
                or getattr(config, "_name_or_path", "")
                or "local-model"
            )

            # Prepend raw sink token x0 to null_states to eliminate inference parasitic drift
            sink_tokens = catcher.get_sink_tokens("o_proj")
            null_states_list: list[np.ndarray] | None = None
            if null_states is not None:
                if isinstance(null_states, dict):
                    raw_nulls = [null_states[idx] for idx in catcher.layer_indices if idx in null_states]
                elif isinstance(null_states, (list, tuple)):
                    raw_nulls = list(null_states)
                else:
                    raw_nulls = []

                null_states_list = []
                for pos, l_idx in enumerate(catcher.layer_indices):
                    sink = sink_tokens.get(l_idx)
                    existing = raw_nulls[pos] if pos < len(raw_nulls) else None
                    if sink is not None and existing is not None:
                        if sink.shape[1] == existing.shape[1]:
                            null_states_list.append(np.vstack([sink, existing]))
                        else:
                            null_states_list.append(existing)
                    elif existing is not None:
                        null_states_list.append(existing)
                    elif sink is not None:
                        null_states_list.append(sink)
            elif sink_tokens:
                # Prepend sink tokens to null states so x0 * Delta W = 0
                null_states_list = [
                    sink_tokens[l_idx] for l_idx in catcher.layer_indices if l_idx in sink_tokens
                ]
                if len(null_states_list) != len(catcher.layer_indices):
                    null_states_list = None

            # depth_scale attenuation removed: it shrank ||ΔW|| by 1/sqrt(L) and
            # caused prior dominance. Amplitude is carried by the fit itself.
            # scale_depth is accepted for API compatibility but has no effect.
            depth_scale_val = 1.0

            self.last_compile_time_ms = compile_all_layers_to_safetensors(
                output_dir=destination,
                rank=rank,
                base_model_name_or_path=base_model_name,
                targets=list(catcher.targets),
                target_inputs=target_inputs,
                target_outputs=target_outputs,
                null_states=null_states_list,
                energy_threshold=energy_threshold,
                min_rank=min_rank,
                max_rank=max_rank,
                max_anchors=max_anchors,
                lambda_reg=lambda_reg,
                lambda_null=lambda_null,
                depth_scale=float(depth_scale_val),
                layer_indices=list(catcher.layer_indices),
            )

            if strict_null_projection:
                from .null_space import project_lora_to_null_space
                from safetensors.torch import load_file, save_file

                sf_path = destination / "adapter_model.safetensors"
                tensors = load_file(str(sf_path))
                o_sinks = catcher.get_sink_tokens("o_proj")
                mlp_sinks = catcher.get_sink_tokens("down_proj")
                for pos, layer_idx in enumerate(catcher.layer_indices):
                    for target_mod, sink_map in (("self_attn.o_proj", o_sinks), ("mlp.down_proj", mlp_sinks)):
                        a_key = f"base_model.model.model.layers.{layer_idx}.{target_mod}.lora_A.weight"
                        b_key = f"base_model.model.model.layers.{layer_idx}.{target_mod}.lora_B.weight"
                        if a_key in tensors and b_key in tensors:
                            target_null: np.ndarray | None = None
                            if layer_idx in sink_map:
                                target_null = sink_map[layer_idx]
                            elif null_states_list is not None and pos < len(null_states_list):
                                if tensors[a_key].shape[1] == null_states_list[pos].shape[1]:
                                    target_null = null_states_list[pos]
                            if target_null is not None:
                                if target_null.ndim == 1:
                                    target_null = target_null.reshape(1, -1)
                                if tensors[a_key].shape[1] == target_null.shape[1]:
                                    a_proj, _ = project_lora_to_null_space(
                                        tensors[a_key], tensors[b_key], target_null
                                    )
                                    tensors[a_key] = a_proj
                save_file(tensors, str(sf_path))
        finally:
            catcher.clear()
            self.model.train(was_training)
            if hasattr(encoded, "clear"):
                encoded.clear()

        expected = (
            destination / "adapter_model.safetensors",
            destination / "adapter_config.json",
        )
        if not all(path.is_file() for path in expected):
            raise RuntimeError(
                f"compilation completed without error but adapter files were not written to {destination}"
            )
        return str(destination)

    def compile_stream(
        self,
        context_text: str | Sequence[int] | torch.Tensor,
        output_dir: str,
        chunk_size: int = 1024,
        rank: int = 16,
        lambda_reg: float = 1e-4,
        overlap_tokens: int = 128,
        null_states: Any = None,
        strict_null_projection: bool = False,
        layer_indices: Iterable[int] | None = None,
        use_rsvd: bool = False,
        **kwargs: Any,
    ) -> str:
        import time
        from .streaming import StreamingCovarianceAccumulator

        start_time = time.perf_counter()
        d_model = getattr(
            self.model,
            "d_model",
            getattr(getattr(self.model, "config", None), "hidden_size", None),
        )
        selected_layers = resolve_layer_indices(self.model, layer_indices)
        accumulator = StreamingCovarianceAccumulator(
            d_model=d_model, layer_indices=selected_layers
        )
        total_tokens = accumulator.accumulate_from_model(
            model=self.model,
            tokenizer=self.tokenizer,
            text_or_tokens=context_text,
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
        )
        self.last_prompt_tokens = total_tokens
        config = getattr(self.model, "config", None)
        base_name = str(
            getattr(self.model, "name_or_path", "")
            or getattr(config, "_name_or_path", "")
            or "local-model"
        )
        out_path = accumulator.save_adapter(
            output_dir=output_dir,
            rank=rank,
            lambda_reg=lambda_reg,
            base_model_name_or_path=base_name,
            method="randomized_svd" if use_rsvd else "svd",
            null_states=null_states,
            strict_null_projection=strict_null_projection,
        )
        self.last_compile_time_ms = (time.perf_counter() - start_time) * 1000.0
        return str(out_path)

    @staticmethod
    def inject_engram(model: nn.Module, adapter_dir: str) -> nn.Module:
        from peft import PeftModel

        adapter_path = Path(adapter_dir).expanduser().resolve()
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter_path / "adapter_model.safetensors")
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter_path / "adapter_config.json")

        if hasattr(model, "unload"):
            try:
                base = model.unload()
            except Exception:
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
        elif hasattr(model, "get_base_model"):
            base = model.get_base_model()
        else:
            base = model
        return PeftModel.from_pretrained(base, str(adapter_path))
