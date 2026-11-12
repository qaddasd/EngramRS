"""End-to-end context-erasure benchmark on a real pretrained model.

Gated behind ENGRAM_RUN_E2E=1 (downloads Qwen2.5-0.5B-Instruct). It validates the
three-tier memory contract honestly:

* Residual-KV tier (query-aware SnapKV) MUST recall a verbatim fact buried at
  ~30% depth after the full source context is erased and replaced by a short
  resident prefix. The question is the observation window; a query-agnostic
  context-tail window drops this needle (verified 0%). This is the "no 0%"
  guarantee.
* Pure-weight LoRA compilation is reported as a diagnostic, NOT asserted: an
  arbitrary verbatim fact cannot be stored in low-rank weights alone (the
  information-theoretic limit documented in REDESIGN_V2). Expect ~0% there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from engram_rs import EngramCompiler, EngramPrefixTuner

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("ENGRAM_RUN_E2E") != "1",
        reason="set ENGRAM_RUN_E2E=1 to run the pretrained-model erasure benchmark",
    ),
]

SECRET_CODE = "428719"
QUESTION = "Какой секретный код запуска? Ответь только числом."
FACT = f"Секретный код запуска: {SECRET_CODE}."
FILLER = (
    "Лабораторный архив Актау, запись Омега-9. Главный архитектор — доктор Кенжегали. "
    "Контрольная фраза хранится только в этом документе. Система мониторинга стабильна. "
)


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


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


def generate_answer(model, inputs: dict, tokenizer, max_new_tokens: int = 24) -> str:
    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return tokenizer.decode(
        generated[0, prompt_tokens:],
        skip_special_tokens=True,
    ).strip()


def make_context() -> str:
    # A SINGLE needle buried at ~30% depth, with a long pure-filler tail. A
    # query-agnostic context-tail window would MISS this needle (verified 0%);
    # only query-aware SnapKV (question as the observation window) retains it.
    block = (FILLER + " ") * 4
    return block * 3 + FACT + " " + block * 7


def test_context_erasure_needle_benchmark(tmp_path: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.environ.get("ENGRAM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    prefix_len = int(os.environ.get("ENGRAM_PREFIX_LEN", "256"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # SnapKV selection needs eager attention to read attention weights.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    context_text = make_context()
    before_tokens = len(tokenizer(context_text, add_special_tokens=False)["input_ids"])

    def chat_ids(content: str) -> dict:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return move_inputs(tokenizer(text, return_tensors="pt"), device)

    # 1. Full-context ceiling: the needle must be copyable when fully visible.
    ceiling_answer = generate_answer(model, chat_ids(context_text + "\n\n" + QUESTION), tokenizer)
    assert SECRET_CODE in ceiling_answer, (
        f"needle is not copyable even with full context: {ceiling_answer!r}"
    )

    # 2. Full KV-cache memory (the cost we are erasing).
    tuner = EngramPrefixTuner(model=model, tokenizer=tokenizer, preserve_sinks=1)
    full_cache = tuner.capture_kv_cache(context_text=context_text)
    kv_mib = cache_nbytes(full_cache) / (1024 * 1024)
    del full_cache

    # 3. Residual-KV tier: query-aware SnapKV prefix replaces the full context.
    # The observation window is the SAME chat-templated question used at
    # generation, so selection scores exactly the tokens the query attends to.
    chat_question = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
    )
    prefix = tuner.compile_prefix(
        context_text=context_text, prefix_len=prefix_len, selection="snapkv",
        query_text=chat_question,
    )
    prefix_mib = cache_nbytes(prefix) / (1024 * 1024)
    question_inputs = chat_ids(QUESTION)
    assert SECRET_CODE not in tokenizer.decode(question_inputs["input_ids"][0])
    after_tokens = int(question_inputs["input_ids"].shape[1])
    prefix_inputs = tuner.prepare_generate_inputs(
        question_inputs["input_ids"], prefix, attention_mask=question_inputs.get("attention_mask")
    )
    engram_answer = generate_answer(model, prefix_inputs, tokenizer)
    recall_percent = 100.0 if SECRET_CODE in engram_answer else 0.0

    # 4. Pure-weight LoRA diagnostic (information-theoretic limit; not asserted).
    compiler = EngramCompiler(model, tokenizer)
    adapter_dir = compiler.compile_text(
        context_text, str(tmp_path / "engram"), rank=16, max_anchors=32, layer_indices=range(8, 16)
    )
    compile_ms = compiler.last_compile_time_ms
    assert compile_ms is not None
    lora_mib = (Path(adapter_dir) / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
    adapted = EngramCompiler.inject_engram(model, adapter_dir)
    lora_answer = generate_answer(adapted, question_inputs, tokenizer)
    lora_recall = 100.0 if SECRET_CODE in lora_answer else 0.0

    print(f"\nFull-context ceiling: {ceiling_answer!r}")
    print(f"Residual-KV (query-aware SnapKV) answer: {engram_answer!r}")
    print(f"Pure-LoRA answer: {lora_answer!r}")
    print("\n| Metric | Value |")
    print("|---|---:|")
    print(f"| Runtime Device | {device} |")
    print(f"| Context Tokens (erased) | {before_tokens} |")
    print(f"| Question Tokens | {after_tokens} |")
    print(f"| Resident Prefix Tokens | {prefix_len} |")
    print(f"| Full KV Cache (MiB) | {kv_mib:.3f} |")
    print(f"| Resident Prefix KV (MiB) | {prefix_mib:.3f} |")
    print(f"| KV Compression (%) | {100.0 * (1.0 - prefix_mib / kv_mib):.1f} |")
    print(f"| LoRA Weights (MiB) | {lora_mib:.3f} |")
    print(f"| Rust Compile Time (ms) | {compile_ms:.3f} |")
    print(f"| Residual-KV Fact Recall (%) | {recall_percent:.1f} |")
    print(f"| Pure-LoRA Fact Recall (%) [diagnostic] | {lora_recall:.1f} |")

    # The residual-KV tier must recall the erased fact and actually compress.
    assert recall_percent == 100.0, f"residual-KV recall failed: {engram_answer!r}"
    assert prefix_mib < kv_mib, "resident prefix did not compress the KV-cache"
    assert after_tokens < before_tokens
    # Sanity gate against pathological regressions (e.g. a debug native build),
    # not a hardware-specific SLA. Real compile time is reported above.
    assert compile_ms < 30000.0, f"Rust compilation took {compile_ms:.3f} ms"
    assert (Path(adapter_dir) / "adapter_model.safetensors").is_file()
