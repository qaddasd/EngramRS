from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from engram_rs import EngramCompiler

pytestmark = [
    pytest.mark.e2e,
]

SECRET_CODE = "7391-KZR"
QUESTION = "Какой секретный код запуска протокола Омега-9?"
ANSWER = f" {SECRET_CODE}"


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_nbytes(cache) -> int:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if isinstance(cache, torch.Tensor):
        return cache.numel() * cache.element_size()
    if isinstance(cache, dict):
        return sum(cache_nbytes(value) for value in cache.values())
    if isinstance(cache, (tuple, list)):
        return sum(cache_nbytes(value) for value in cache)
    if hasattr(cache, "layers"):
        return sum(cache_nbytes(layer) for layer in cache.layers)
    keys = getattr(cache, "keys", None)
    values = getattr(cache, "values", None)
    total = 0
    if keys is not None:
        total += cache_nbytes(keys)
    if values is not None:
        total += cache_nbytes(values)
    if total:
        return total
    raise TypeError(f"unsupported KV-cache representation: {type(cache).__name__}")


def measure_ttft(model, inputs: dict, device: torch.device) -> float:
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def target_perplexity(model, prompt: str, answer: str, tokenizer, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.shape[1] == 0:
        raise ValueError("target answer produced no tokens")
    input_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    return torch.exp(loss).item()


def generate_answer(model, inputs: dict, tokenizer) -> str:
    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return tokenizer.decode(
        generated[0, prompt_tokens:],
        skip_special_tokens=True,
    ).strip()


def make_context(tokenizer, target_tokens: int) -> str:
    paragraph = (
        "Лабораторный архив Актау, запись Омега-9.\n"
        f"{QUESTION}\nОтвет:{ANSWER}.\n"
        f"Секретный код запуска протокола Омега-9 —{ANSWER}; "
        "главный архитектор — доктор Кенжегали. "
        "Контрольная фраза хранится только в этом документе.\n"
    )
    context = paragraph
    while len(tokenizer(context, add_special_tokens=False)["input_ids"]) < target_tokens:
        context += paragraph
    return context


def test_context_erasure_needle_benchmark(tmp_path: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.environ.get("ENGRAM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    target_tokens = int(os.environ.get("ENGRAM_CONTEXT_TOKENS", "256"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    context_text = make_context(tokenizer, target_tokens)
    compiler = EngramCompiler(model, tokenizer)
    adapter_dir = compiler.compile_text(
        context_text, str(tmp_path / "engram"), rank=16, max_anchors=32, layer_indices=range(8, 16)
    )
    compile_ms = compiler.last_compile_time_ms
    assert compile_ms is not None
    before_tokens = compiler.last_prompt_tokens
    assert before_tokens is not None

    context_inputs = move_inputs(tokenizer(context_text, return_tensors="pt"), device)
    with torch.no_grad():
        backbone = getattr(model, "model", model)
        context_output = backbone(**context_inputs, use_cache=True)
    kv_mib = cache_nbytes(context_output.past_key_values) / (1024 * 1024)
    del context_output, context_inputs, context_text
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prompt = QUESTION + "\nОтвет:"
    assert SECRET_CODE not in prompt
    question_inputs = move_inputs(tokenizer(prompt, return_tensors="pt"), device)
    after_tokens = int(question_inputs["input_ids"].shape[1])
    lora_mib = (Path(adapter_dir) / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
    adapted = EngramCompiler.inject_engram(model, adapter_dir)

    with adapted.disable_adapter():
        base_ttft_ms = measure_ttft(adapted, question_inputs, device)
        base_perplexity = target_perplexity(adapted, prompt, ANSWER, tokenizer, device)
        base_answer = generate_answer(adapted, question_inputs, tokenizer)
    engram_ttft_ms = measure_ttft(adapted, question_inputs, device)
    engram_perplexity = target_perplexity(adapted, prompt, ANSWER, tokenizer, device)
    engram_answer = generate_answer(adapted, question_inputs, tokenizer)
    recall_percent = 100.0 if SECRET_CODE in engram_answer else 0.0
    perplexity_drop = (base_perplexity - engram_perplexity) / base_perplexity * 100.0

    print(f"\nBase answer: {base_answer!r}")
    print(f"Engram answer: {engram_answer!r}")
    print("\n| Metric | Value |")
    print("|---|---:|")
    print(f"| Runtime Device | {device} |")
    print(f"| Compile Time (ms) | {compile_ms:.3f} |")
    print(f"| Prompt Tokens Before | {before_tokens} |")
    print(f"| Prompt Tokens After | {after_tokens} |")
    print(f"| KV Cache Memory (MiB) | {kv_mib:.3f} |")
    print(f"| LoRA Weights (MiB) | {lora_mib:.3f} |")
    print(f"| Base TTFT (ms) | {base_ttft_ms:.3f} |")
    print(f"| Engram TTFT (ms) | {engram_ttft_ms:.3f} |")
    print(f"| Fact Recall Accuracy (%) | {recall_percent:.1f} |")
    print(f"| Base Target Perplexity | {base_perplexity:.6f} |")
    print(f"| Engram Target Perplexity | {engram_perplexity:.6f} |")
    print(f"| Perplexity Drop (%) | {perplexity_drop:.3f} |")

    assert compile_ms < 1000.0, f"Rust compilation took {compile_ms:.3f} ms"
    assert after_tokens < before_tokens
    assert (Path(adapter_dir) / "adapter_model.safetensors").is_file()
