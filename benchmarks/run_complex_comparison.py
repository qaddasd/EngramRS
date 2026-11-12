"""Comprehensive Deep Benchmark: Model Performance With and Without EngramRS.

Compares:
1. Standard In-Context Prompting (Traditional KV-cache, Full Context)
2. Zero Context Baseline (Context erased, no Engram adapter)
3. EngramRS LoRA Memory Injection (Context erased, LoRA adapter active)
4. EngramRS Query-Aware KV-Prefix (Context erased into a per-question SnapKV prefix)
5. Prefix Caching Reference (vLLM / SGLang concept: KV retained in VRAM)

Evaluates:
- TTFT (Time-To-First-Token prefill latency in ms, median of multiple runs)
- Prompt token compression ratio
- KV-Cache VRAM usage (MiB, measured for both full and bare queries)
- Target fact perplexity
- Exact factual recall accuracy (%) across 5 distinct domains
- Rust compilation latency (ms)
- Scaling across context lengths (512, 1024, 2048 tokens)
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram_rs import EngramCompiler, EngramPrefixTuner
from transformers.cache_utils import DynamicCache

# Resident prefix budget for the residual-KV tier. ~12% of a 2k-token context;
# large enough for SnapKV to retain fact-bearing tokens (see REDESIGN_V2 §2.4).
PREFIX_LEN = 256


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_nbytes(cache: Any) -> int:
    """Measure exact memory in bytes occupied by a KV cache structure."""
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if isinstance(cache, torch.Tensor):
        return cache.numel() * cache.element_size()
    if isinstance(cache, dict):
        return sum(cache_nbytes(v) for v in cache.values())
    if isinstance(cache, (tuple, list)):
        return sum(cache_nbytes(v) for v in cache)
    if hasattr(cache, "layers"):
        return sum(cache_nbytes(l) for l in cache.layers)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return sum(cache_nbytes(k) for k in cache.key_cache) + sum(
            cache_nbytes(v) for v in cache.value_cache
        )
    return 0


def get_base_model(model_instance: Any) -> Any:
    """Ensure model is unwrapped from any PEFT adapter layer."""
    if hasattr(model_instance, "unload"):
        try:
            return model_instance.unload()
        except Exception:
            pass
    if hasattr(model_instance, "get_base_model"):
        try:
            return model_instance.get_base_model()
        except Exception:
            pass
    return model_instance


def measure_ttft(
    model: Any,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    warmup_runs: int = 2,
    measured_runs: int = 5,
) -> float:
    """Measure Time-To-First-Token prefill latency in ms via median of multiple synchronized runs."""
    synchronize(device)

    def run_once() -> None:
        run_inputs = dict(inputs)
        pkv = run_inputs.get("past_key_values")
        if pkv is not None:
            # Clone the cache so repeated runs do not accumulate generated tokens.
            legacy = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
            run_inputs["past_key_values"] = DynamicCache.from_legacy_cache(
                tuple((k.clone(), v.clone()) for k, v in legacy)
            )
        with torch.no_grad():
            model(**run_inputs)

    for _ in range(warmup_runs):
        run_once()
    synchronize(device)

    latencies: list[float] = []
    for _ in range(measured_runs):
        synchronize(device)
        t0 = time.perf_counter()
        run_once()
        synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    return latencies[len(latencies) // 2]


def compute_target_perplexity(
    model: Any,
    prompt: str,
    target_answer: str,
    tokenizer: Any,
    device: torch.device,
) -> float:
    """Compute perplexity strictly on the target answer tokens given the prompt."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(target_answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.shape[1] == 0:
        return 999.0
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
    return float(torch.exp(loss).item())


def generate_text(
    model: Any,
    inputs: dict[str, torch.Tensor],
    tokenizer: Any,
    max_new_tokens: int = 64,
) -> tuple[str, float]:
    """Generate tokens and record total generation latency in ms."""
    device = inputs["input_ids"].device
    prompt_len = inputs["input_ids"].shape[1]
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    generated_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True).strip()
    return generated_text, elapsed_ms


# ============================================================================
# TECHNICAL HAYSTACK & TEST NEEDLES ACROSS 5 DIVERSE DOMAINS
# ============================================================================

TECHNICAL_DOMAINS = [
    {
        "domain": "Cryptography",
        "needle": "CRITICAL SECURITY NOTICE: The master encryption override code for Sector 7 is CIPHER-7749-DELTA.",
        "question": "What is the master encryption override code for Sector 7?",
        "expected": "CIPHER-7749-DELTA",
        "depth": 0.15,
    },
    {
        "domain": "Distributed Databases",
        "needle": "DATABASE TOPOLOGY: The primary coordinator node in cluster Atlas is located at IP address 172.28.94.215.",
        "question": "What is the IP address of the primary coordinator node in cluster Atlas?",
        "expected": "172.28.94.215",
        "depth": 0.35,
    },
    {
        "domain": "Quantum Hardware",
        "needle": "HARDWARE SPECIFICATION: The operational qubit coherence dephasing time T2 is exactly 142.6 microseconds.",
        "question": "What is the operational qubit coherence dephasing time T2?",
        "expected": "142.6 microseconds",
        "depth": 0.55,
    },
    {
        "domain": "Aerospace Engineering",
        "needle": "PROPULSION TELEMETRY: The specific impulse achieved by the ion thruster in vacuum test vacuum-4 is 3850 seconds.",
        "question": "What is the specific impulse achieved by the ion thruster in vacuum test vacuum-4?",
        "expected": "3850 seconds",
        "depth": 0.75,
    },
    {
        "domain": "Biochemical Synthesis",
        "needle": "ENZYMATIC PROTOCOL: The optimal thermal stabilization buffer pH for protein variant K-42 is 6.85.",
        "question": "What is the optimal thermal stabilization buffer pH for protein variant K-42?",
        "expected": "6.85",
        "depth": 0.95,
    },
]

BACKGROUND_PARAGRAPHS = [
    (
        "Modern distributed consensus protocols rely on state machine replication. "
        "Under the Paxos algorithm, nodes exchange prepare and accept phases to achieve quorum. "
        "Raft simplifies leader election by utilizing randomized election timers and explicit term numbers. "
        "Network partitions are mitigated through majority heartbeat verification."
    ),
    (
        "Quantum computing platforms utilize superconducting qubits cooled to millikelvin temperatures. "
        "Josephson junctions provide non-linear inductance necessary to isolate computational energy levels. "
        "Decoherence times are bounded by dielectric loss and magnetic flux fluctuations in the substrate. "
        "Surface codes perform continuous syndrome measurements for fault-tolerant error correction."
    ),
    (
        "High-performance numerical linear algebra achieves near-peak FLOP utilization through cache blocking. "
        "Matrix multiplication routines divide operands into sub-matrices fitting in L1, L2, and L3 caches. "
        "Vectorized SIMD instruction sets such as AVX-512 and ARM Neon execute parallel multiply-accumulate operations. "
        "Strassen's algorithm reduces asymptotic complexity for very large matrices."
    ),
    (
        "Cellular biology investigates membrane transport proteins that maintain electrochemical gradients. "
        "The sodium-potassium adenosine triphosphatase pump moves three sodium ions out for every two potassium ions in. "
        "Action potentials propagate along myelinated axons through saltatory conduction between nodes of Ranvier. "
        "Synaptic vesicles fuse with the presynaptic terminal upon calcium ion influx."
    ),
    (
        "Geological plate tectonics explain continental drift and subduction zone volcanism. "
        "Convection currents within the asthenosphere drive the movement of rigid lithospheric plates. "
        "Mid-ocean ridges produce new basaltic oceanic crust through seafloor spreading. "
        "Transform faults exhibit strike-slip motion responsible for shallow focus seismic events."
    ),
    (
        "Photolithography techniques in semiconductor manufacturing utilize extreme ultraviolet light at 13.5 nm wavelength. "
        "Reflective multilayer mirrors composed of molybdenum and silicon focus the beam onto wafer targets. "
        "Chemical amplification in photoresists generates acid catalysts that cleave protective polymer groups during baking. "
        "Immersion systems historically improved numerical aperture using purified water layers."
    ),
]


def build_multi_needle_context(
    target_tokens: int,
    needles: list[dict[str, Any]],
    tokenizer: Any,
) -> str:
    """Build a coherent background context embedding all needles at their respective fractional depths."""
    paragraphs: list[str] = []
    idx = 0
    while len(tokenizer.encode(" ".join(paragraphs))) < target_tokens:
        paragraphs.append(BACKGROUND_PARAGRAPHS[idx % len(BACKGROUND_PARAGRAPHS)])
        idx += 1

    sorted_needles = sorted(needles, key=lambda n: n["depth"], reverse=True)
    total_paras = len(paragraphs)
    for item in sorted_needles:
        pos = int(round(total_paras * item["depth"]))
        paragraphs.insert(pos, item["needle"])

    return "\n\n".join(paragraphs)


def run_benchmark() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n{'='*80}")
    print(f"ENGRAM-RS DEEP BENCHMARK & COMPARISON SUITE (v0.3.0)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Target Model: {model_name} (dtype: {dtype})")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Warmup
    print("Warmup model on device...")
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        base_model(warmup_ids)
    synchronize(device)

    # =========================================================================
    # PART 1: MULTI-NEEDLE FACT RETRIEVAL (2000 TOKENS)
    # =========================================================================
    print(f"\n--- PART 1: 5-DOMAIN MULTI-NEEDLE BENCHMARK (~2000 TOKENS) ---")
    context_2k = build_multi_needle_context(1800, TECHNICAL_DOMAINS, tokenizer)
    context_tokens = len(tokenizer.encode(context_2k))
    print(f"Built realistic technical context with {context_tokens} tokens and 5 needles.\n")

    # Measure KV Cache for full context
    context_input = tokenizer(context_2k, return_tensors="pt")
    context_input = {k: v.to(device) for k, v in context_input.items()}
    with torch.no_grad():
        backbone = getattr(base_model, "model", base_model)
        c_out = backbone(**context_input, use_cache=True)
        kv_cache_bytes = cache_nbytes(c_out.past_key_values)
        kv_cache_mib = kv_cache_bytes / (1024 * 1024)
    del c_out, context_input
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Measure KV Cache for short bare query (~16 tokens)
    sample_bare_prompt = f"Question: {TECHNICAL_DOMAINS[0]['question']}\nAnswer:"
    sample_bare_input = {k: v.to(device) for k, v in tokenizer(sample_bare_prompt, return_tensors="pt").items()}
    with torch.no_grad():
        b_bare_out = backbone(**sample_bare_input, use_cache=True)
        bare_kv_bytes = cache_nbytes(b_bare_out.past_key_values)
        bare_kv_mib = bare_kv_bytes / (1024 * 1024)
    del b_bare_out, sample_bare_input
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Compile with EngramRS from clean base model
    with tempfile.TemporaryDirectory() as tmp_dir:
        compiler = EngramCompiler(base_model, tokenizer)
        print("Compiling context into Engram LoRA adapter in Rust...")
        adapter_path = compiler.compile_text(
            context_2k,
            output_dir=tmp_dir,
            rank=16,
            max_anchors=32,
            layer_indices=range(8, 16),
            targets=("o_proj", "down_proj"),
        )
        compile_time_ms = compiler.last_compile_time_ms or 0.0
        adapter_bytes = (Path(adapter_path) / "adapter_model.safetensors").stat().st_size
        adapter_mib = adapter_bytes / (1024 * 1024)
        print(f"Compilation finished in {compile_time_ms:.2f} ms! Adapter size: {adapter_mib:.3f} MiB\n")

        # Residual-KV tier: a SnapKV prefix tuner on the CLEAN base model
        # (before LoRA injection) so the prefix reflects base activations.
        # The prefix is compiled PER DOMAIN with the question as the SnapKV
        # observation window (query-aware). A fixed query-agnostic prefix drops
        # buried needles (verified 0/5); query-aware selection retains them.
        tuner = EngramPrefixTuner(
            model=base_model,
            tokenizer=tokenizer,
            prefix_len=PREFIX_LEN,
            preserve_sinks=1,
        )
        print(f"Query-aware SnapKV prefix: {PREFIX_LEN} tokens/domain "
              f"(compression ratio: {context_tokens / PREFIX_LEN:.1f}x)\n")

        # Inject adapter into a clean copy/wrapper
        adapted_model = EngramCompiler.inject_engram(base_model, adapter_path)

        # Evaluate across the 5 domains
        comparison_results = []
        full_success_count = 0
        bare_success_count = 0
        engram_success_count = 0
        prefix_success_count = 0

        for item in TECHNICAL_DOMAINS:
            domain = item["domain"]
            question = item["question"]
            expected = item["expected"]
            depth_pct = f"{int(item['depth']*100)}%"

            # 1. Full In-Context (Traditional Prompting on clean base model)
            full_prompt = f"{context_2k}\n\nQuestion: {question}\nAnswer:"
            full_inputs = {k: v.to(device) for k, v in tokenizer(full_prompt, return_tensors="pt").items()}
            full_tokens = full_inputs["input_ids"].shape[1]

            with adapted_model.disable_adapter():
                full_ttft = measure_ttft(adapted_model, full_inputs, device)
                full_ppl = compute_target_perplexity(adapted_model, full_prompt, f" {expected}", tokenizer, device)
                full_gen, full_lat = generate_text(adapted_model, full_inputs, tokenizer)
                full_success = expected.lower() in full_gen.lower()
                if full_success:
                    full_success_count += 1

            # 2. Baseline Erased (No Context, No Engram on clean base model)
            bare_prompt = f"Question: {question}\nAnswer:"
            bare_inputs = {k: v.to(device) for k, v in tokenizer(bare_prompt, return_tensors="pt").items()}
            bare_tokens = bare_inputs["input_ids"].shape[1]

            with adapted_model.disable_adapter():
                bare_ttft = measure_ttft(adapted_model, bare_inputs, device)
                bare_ppl = compute_target_perplexity(adapted_model, bare_prompt, f" {expected}", tokenizer, device)
                bare_gen, bare_lat = generate_text(adapted_model, bare_inputs, tokenizer)
                bare_success = expected.lower() in bare_gen.lower()
                if bare_success:
                    bare_success_count += 1

            # 3. EngramRS (Context Erased, LoRA Active)
            engram_ttft = measure_ttft(adapted_model, bare_inputs, device)
            engram_ppl = compute_target_perplexity(adapted_model, bare_prompt, f" {expected}", tokenizer, device)
            engram_gen, engram_lat = generate_text(adapted_model, bare_inputs, tokenizer)
            engram_success = expected.lower() in engram_gen.lower()
            if engram_success:
                engram_success_count += 1

            ppl_drop = (bare_ppl - engram_ppl) / bare_ppl * 100.0

            # 4. Query-aware SnapKV prefix (context erased, base model).
            # The question is the observation window, so fact-bearing tokens are
            # retained. prepare_generate_inputs sets cache_position = arange(B, B+q)
            # and a prefix-covering attention_mask; a fresh cloned cache avoids
            # generate() mutating the prefix across repeated runs.
            with adapted_model.disable_adapter():
                domain_prefix = tuner.compile_prefix(
                    context_text=context_2k,
                    prefix_len=PREFIX_LEN,
                    selection="snapkv",
                    query_text=f"Question: {question}\nAnswer:",
                )
                prefix_legacy = tuple(
                    (k.clone(), v.clone())
                    for k, v in (domain_prefix[i] for i in range(len(domain_prefix)))
                )
                del domain_prefix

                def fresh_prefix_inputs() -> dict:
                    cache = DynamicCache.from_legacy_cache(
                        tuple((k.clone(), v.clone()) for k, v in prefix_legacy)
                    )
                    return tuner.prepare_generate_inputs(
                        bare_inputs["input_ids"],
                        cache,
                        attention_mask=bare_inputs.get("attention_mask"),
                    )

                prefix_ttft = measure_ttft(adapted_model, fresh_prefix_inputs(), device)
                prefix_gen, prefix_lat = generate_text(adapted_model, fresh_prefix_inputs(), tokenizer)
                prefix_success = expected.lower() in prefix_gen.lower()
                if prefix_success:
                    prefix_success_count += 1

            comparison_results.append({
                "domain": domain,
                "depth": depth_pct,
                "expected": expected,
                "full": {
                    "tokens": full_tokens,
                    "ttft_ms": full_ttft,
                    "ppl": full_ppl,
                    "gen": full_gen,
                    "success": full_success,
                },
                "bare": {
                    "tokens": bare_tokens,
                    "ttft_ms": bare_ttft,
                    "ppl": bare_ppl,
                    "gen": bare_gen,
                    "success": bare_success,
                },
                "engram": {
                    "tokens": bare_tokens,
                    "ttft_ms": engram_ttft,
                    "ppl": engram_ppl,
                    "ppl_drop": ppl_drop,
                    "gen": engram_gen,
                    "success": engram_success,
                },
                "prefix": {
                    "tokens": PREFIX_LEN + bare_tokens,
                    "ttft_ms": prefix_ttft,
                    "gen": prefix_gen,
                    "success": prefix_success,
                },
            })

        # Clean unwrap adapted model
        if hasattr(adapted_model, "unload"):
            base_model = adapted_model.unload()

    # Print Table 1: Multi-Needle Per-Domain Results
    print("=" * 125)
    print(f"{'DOMAIN':<22} | {'DEPTH':<6} | {'FULL CTX PPL':<12} | {'NO CTX PPL':<12} | {'ENGRAM PPL':<12} | {'PPL DROP (%)':<12} | {'TTFT BOOST':<10} | {'RECALL'}")
    print("-" * 125)
    for r in comparison_results:
        f = r["full"]
        b = r["bare"]
        e = r["engram"]
        ttft_speedup = f"{f['ttft_ms'] / e['ttft_ms']:.1f}x"
        recall_tag = "1/1" if e["success"] else "0/1"
        print(f"{r['domain']:<22} | {r['depth']:<6} | {f['ppl']:<12.3f} | {b['ppl']:<12.3f} | {e['ppl']:<12.3f} | {e['ppl_drop']:<12.2f}% | {ttft_speedup:<10} | {recall_tag}")
    print("=" * 125)

    print("\nEXACT FACT RECALL ACCURACY SUMMARY:")
    print(f"  • Full Context (KV {context_tokens}t) : {full_success_count} / {len(TECHNICAL_DOMAINS)} ({full_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)")
    print(f"  • Without Context (Base)   : {bare_success_count} / {len(TECHNICAL_DOMAINS)} ({bare_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)")
    print(f"  • With Engram (LoRA)       : {engram_success_count} / {len(TECHNICAL_DOMAINS)} ({engram_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)")
    print(f"  • With Query-Aware Prefix ({PREFIX_LEN}t): {prefix_success_count} / {len(TECHNICAL_DOMAINS)} ({prefix_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)")

    print("\nSAMPLE GENERATIONS COMPARISON:")
    for r in comparison_results:
        print(f"\n[Domain: {r['domain']} | Expected: {r['expected']}]")
        print(f"  • Without Context (Base)   : {r['bare']['gen']!r}")
        print(f"  • With Engram (LoRA)       : {r['engram']['gen']!r}")
        print(f"  • With Query-Aware Prefix ({PREFIX_LEN}t): {r['prefix']['gen']!r}")
        print(f"  • Full Context (KV 2000t)  : {r['full']['gen']!r}")

    # =========================================================================
    # PART 2: CONTEXT LENGTH SCALING BENCHMARK (512, 1024, 2048 TOKENS)
    # =========================================================================
    print(f"\n\n--- PART 2: CONTEXT LENGTH SCALING BENCHMARK (512, 1024, 2048 TOKENS) ---")
    scaling_lengths = [512, 1024, 2048]
    scaling_data = []

    for c_len in scaling_lengths:
        # Guarantee clean un-adapted model
        base_model = get_base_model(base_model)
        backbone = getattr(base_model, "model", base_model)

        ctx = build_multi_needle_context(c_len, TECHNICAL_DOMAINS[:2], tokenizer)
        actual_tokens = len(tokenizer.encode(ctx))

        # 1. Full context KV cache and TTFT
        ctx_inputs = {k: v.to(device) for k, v in tokenizer(ctx, return_tensors="pt").items()}
        with torch.no_grad():
            b_out = backbone(**ctx_inputs, use_cache=True)
            kv_bytes = cache_nbytes(b_out.past_key_values)
            kv_mib = kv_bytes / (1024 * 1024)
        del b_out, ctx_inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        full_q_prompt = f"{ctx}\n\nQuestion: {TECHNICAL_DOMAINS[0]['question']}\nAnswer:"
        full_q_inputs = {k: v.to(device) for k, v in tokenizer(full_q_prompt, return_tensors="pt").items()}
        full_ttft = measure_ttft(base_model, full_q_inputs, device)

        # 2. EngramRS compilation and inference
        with tempfile.TemporaryDirectory() as s_dir:
            c_inst = EngramCompiler(base_model, tokenizer)
            c_inst.compile_text(
                ctx,
                output_dir=s_dir,
                rank=16,
                max_anchors=32,
                layer_indices=range(8, 16),
            )
            c_ms = c_inst.last_compile_time_ms or 0.0

            q_bare_prompt = f"Question: {TECHNICAL_DOMAINS[0]['question']}\nAnswer:"
            q_bare_inputs = {k: v.to(device) for k, v in tokenizer(q_bare_prompt, return_tensors="pt").items()}
            bare_tokens = q_bare_inputs["input_ids"].shape[1]

            engram_m = EngramCompiler.inject_engram(base_model, s_dir)
            engram_ttft = measure_ttft(engram_m, q_bare_inputs, device)

            # Clean unwrap
            if hasattr(engram_m, "unload"):
                base_model = engram_m.unload()

        speedup = full_ttft / engram_ttft
        tokens_saved = actual_tokens - bare_tokens
        compression_pct = (tokens_saved / actual_tokens) * 100.0

        scaling_data.append({
            "target_len": c_len,
            "actual_tokens": actual_tokens,
            "kv_cache_mib": kv_mib,
            "engram_kv_cache_mib": bare_kv_mib,
            "rust_compile_ms": c_ms,
            "full_ttft_ms": full_ttft,
            "engram_ttft_ms": engram_ttft,
            "speedup": speedup,
            "compression_pct": compression_pct,
        })

    print("=" * 125)
    print(f"{'CONTEXT LEN':<12} | {'FULL KV (MiB)':<14} | {'ENGRAM KV (MiB)':<16} | {'RUST COMPILE':<14} | {'FULL TTFT':<12} | {'ENGRAM TTFT':<12} | {'SPEEDUP':<10} | {'PROMPT COMPR'}")
    print("-" * 125)
    for s in scaling_data:
        print(f"{s['actual_tokens']:<12} | {s['kv_cache_mib']:<14.2f} | {s['engram_kv_cache_mib']:<16.2f} | {s['rust_compile_ms']:<12.1f} ms | {s['full_ttft_ms']:<10.2f} ms | {s['engram_ttft_ms']:<10.2f} ms | {s['speedup']:<8.1f}x | {s['compression_pct']:.1f}%")
    print("=" * 125)

    # =========================================================================
    # PART 3: ARCHITECTURE ABLATION (ATTENTION vs MLP vs DUAL)
    # =========================================================================
    print(f"\n\n--- PART 3: ARCHITECTURAL ABLATION: ATTENTION vs MLP vs DUAL (2000 TOKENS) ---")
    ablation_modes = [
        ("Attention only (`o_proj`)", ("o_proj",)),
        ("MLP SwiGLU only (`down_proj`)", ("down_proj",)),
        ("Dual (`o_proj` + `down_proj`)", ("o_proj", "down_proj")),
    ]

    ablation_results = []
    sample_needle = TECHNICAL_DOMAINS[0]
    eval_prompt = f"Question: {sample_needle['question']}\nAnswer:"
    eval_answer = f" {sample_needle['expected']}"

    # Base without adapter perplexity on clean model
    base_model = get_base_model(base_model)
    base_ppl = compute_target_perplexity(base_model, eval_prompt, eval_answer, tokenizer, device)

    for label, targets in ablation_modes:
        base_model = get_base_model(base_model)
        with tempfile.TemporaryDirectory() as a_dir:
            c = EngramCompiler(base_model, tokenizer)
            c.compile_text(
                context_2k,
                output_dir=a_dir,
                rank=16,
                max_anchors=32,
                layer_indices=range(8, 16),
                targets=targets,
            )
            c_time = c.last_compile_time_ms or 0.0
            adapter_size_kb = (Path(a_dir) / "adapter_model.safetensors").stat().st_size / 1024.0

            ad_model = EngramCompiler.inject_engram(base_model, a_dir)
            eng_ppl = compute_target_perplexity(ad_model, eval_prompt, eval_answer, tokenizer, device)
            ppl_drop = (base_ppl - eng_ppl) / base_ppl * 100.0

            ablation_results.append({
                "label": label,
                "targets": "+".join(targets),
                "compile_ms": c_time,
                "size_kb": adapter_size_kb,
                "ppl": eng_ppl,
                "ppl_drop": ppl_drop,
            })

            if hasattr(ad_model, "unload"):
                base_model = ad_model.unload()

    print("=" * 95)
    print(f"{'TARGET ARCHITECTURE':<32} | {'COMPILE (ms)':<14} | {'SIZE (KB)':<12} | {'PERPLEXITY':<12} | {'PPL DROP'}")
    print("-" * 95)
    print(f"{'Baseline (No Adapter)':<32} | {'—':<14} | {'—':<12} | {base_ppl:<12.3f} | 0.00%")
    for a in ablation_results:
        print(f"{a['label']:<32} | {a['compile_ms']:<12.1f} ms | {a['size_kb']:<10.1f} KB | {a['ppl']:<12.3f} | {a['ppl_drop']:.2f}%")
    print("=" * 95)

    # Save full results as JSON
    results_payload = {
        "device": str(device),
        "model": model_name,
        "exact_recall_summary": {
            "full_context": f"{full_success_count}/{len(TECHNICAL_DOMAINS)} ({full_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)",
            "zero_context": f"{bare_success_count}/{len(TECHNICAL_DOMAINS)} ({bare_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)",
            "engram_lora": f"{engram_success_count}/{len(TECHNICAL_DOMAINS)} ({engram_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)",
            "virtual_prefix": f"{prefix_success_count}/{len(TECHNICAL_DOMAINS)} ({prefix_success_count/len(TECHNICAL_DOMAINS)*100:.1f}%)",
        },
        "part1_multi_needle": comparison_results,
        "part2_scaling": scaling_data,
        "part3_ablation": ablation_results,
    }
    out_json = Path(__file__).parent / "complex_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed metrics saved to: {out_json}")
    print("\nBENCHMARK SUITE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_benchmark()
