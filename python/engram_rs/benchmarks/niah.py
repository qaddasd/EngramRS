"""Needle In A Haystack (NIAH) Benchmark Runner for EngramRS.

Validates fact retention and retrieval from context compiled into Engram adapters
across varying context lengths (e.g., 1k, 4k, 8k tokens) and insertion depths
(0%, 25%, 50%, 75%, 100%).

Unlike standard in-context RAG evaluation where the whole haystack is sent in every prompt,
EngramRS compiles the haystack context once into an adapter. The model is then queried
with ONLY the short retrieval question (empty haystack prompt), validating genuine weight-space
memory consolidation.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from engram_rs.compiler import EngramCompiler


DEFAULT_HAYSTACK_PARAGRAPHS = [
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
]

DEFAULT_NEEDLE = "The secret access passcode to the central vault is 92481-OMEGA."
DEFAULT_QUESTION = "What is the secret access passcode to the central vault?"
DEFAULT_EXPECTED = "92481-OMEGA"


def parse_context_lens(context_lens_arg: str | list[int] | list[str]) -> list[int]:
    """Parse context lengths like '1k,4k,8k' or [1000, 4000, 8000]."""
    if isinstance(context_lens_arg, (list, tuple)):
        result = []
        for item in context_lens_arg:
            if isinstance(item, int):
                result.append(item)
            else:
                s = str(item).strip().lower()
                if s.endswith("k"):
                    result.append(int(float(s[:-1]) * 1000))
                else:
                    result.append(int(s))
        return result

    raw = str(context_lens_arg).split(",")
    lens = []
    for part in raw:
        p = part.strip().lower()
        if not p:
            continue
        if p.endswith("k"):
            lens.append(int(float(p[:-1]) * 1000))
        elif p.endswith("m"):
            lens.append(int(float(p[:-1]) * 1_000_000))
        else:
            lens.append(int(p))
    return lens or [1000, 4000, 8000]


def parse_depths(depths_arg: str | list[float] | list[int]) -> list[float]:
    """Parse depths like '0,25,50,75,100' or [0.0, 0.25, 0.5, 0.75, 1.0]."""
    if isinstance(depths_arg, (list, tuple)):
        result = []
        for item in depths_arg:
            val = float(item)
            if val > 1.0:
                val /= 100.0
            result.append(val)
        return result

    raw = str(depths_arg).split(",")
    depths = []
    for part in raw:
        p = part.strip()
        if not p:
            continue
        val = float(p)
        if val > 1.0:
            val /= 100.0
        depths.append(val)
    return depths or [0.0, 0.25, 0.5, 0.75, 1.0]


def build_haystack_context(
    target_tokens: int,
    needle: str,
    depth: float,
    tokenizer: Any = None,
) -> tuple[str, int]:
    """Generate background text and insert needle at the specified fractional depth.

    Args:
        target_tokens: Approximate target token count.
        needle: Fact string to insert.
        depth: Insertion depth in [0.0, 1.0].
        tokenizer: Optional tokenizer to measure exact token counts.

    Returns:
        (full_context_text, token_count)
    """
    paragraphs = []
    current_tokens = 0
    idx = 0

    # Estimate words per paragraph
    while current_tokens < target_tokens:
        p = DEFAULT_HAYSTACK_PARAGRAPHS[idx % len(DEFAULT_HAYSTACK_PARAGRAPHS)]
        paragraphs.append(p)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                current_tokens = len(tokenizer.encode(" ".join(paragraphs)))
            except Exception:
                current_tokens += len(p.split())
        else:
            # Word-based approximation
            current_tokens += int(len(p.split()) * 1.3)
        idx += 1

    # Place needle at depth
    insert_pos = int(round(len(paragraphs) * max(0.0, min(1.0, depth))))
    paragraphs.insert(insert_pos, needle)

    full_text = "\n\n".join(paragraphs)
    final_tokens = current_tokens
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            final_tokens = len(tokenizer.encode(full_text))
        except Exception:
            pass

    return full_text, final_tokens


def evaluate_retrieval(
    model: nn.Module,
    tokenizer: Any,
    question: str,
    expected_answer: str,
    max_new_tokens: int = 32,
) -> tuple[bool, str, float]:
    """Run model inference on the question prompt and check for the answer."""
    device = next(model.parameters()).device
    start_time = time.perf_counter()

    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    generated_text = ""
    try:
        if hasattr(model, "generate"):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
                    do_sample=False,
                )
            generated_ids = outputs[0][input_ids.shape[-1] :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            # Greedy stepping fallback
            curr_ids = input_ids
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    out = model(input_ids=curr_ids)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
            generated_text = tokenizer.decode(curr_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)
    except Exception as e:
        # If generation fails due to mock / tiny vocabulary limits, check logits or error
        generated_text = f"<eval_error: {e}>"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    success = expected_answer.lower() in generated_text.lower()
    return success, generated_text.strip(), elapsed_ms


def run_niah_benchmark(
    model: nn.Module | str,
    tokenizer: Any = None,
    context_lens: str | list[int] = "1k,4k,8k",
    depths: str | list[float] = "0,25,50,75,100",
    needle: str = DEFAULT_NEEDLE,
    question: str = DEFAULT_QUESTION,
    expected_answer: str = DEFAULT_EXPECTED,
    rank: int = 16,
    evaluator: Callable[..., tuple[bool, str, float]] | None = None,
) -> dict[str, Any]:
    """Execute complete Needle In A Haystack benchmark using Engram compilation.

    For every (context_len, depth) combination:
    1. Haystack context + needle is generated.
    2. Context is compiled into an Engram adapter in zero-copy Rust.
    3. Model is injected with adapter.
    4. Model is queried with ONLY the short question.
    5. Fact retrieval accuracy and latency are recorded.
    """
    if isinstance(model, str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = model
        tokenizer = AutoTokenizer.from_pretrained(model_path) if tokenizer is None else tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)

    parsed_lens = parse_context_lens(context_lens)
    parsed_depths = parse_depths(depths)

    results_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    total_tests = 0
    passed_tests = 0
    compile_times: list[float] = []
    eval_times: list[float] = []

    compiler = EngramCompiler(model, tokenizer)

    for c_len in parsed_lens:
        c_key = f"{c_len}" if c_len < 1000 else f"{c_len // 1000}k"
        results_matrix[c_key] = {}
        for depth in parsed_depths:
            d_key = f"{int(round(depth * 100))}%"

            full_context, actual_tokens = build_haystack_context(
                c_len, needle=needle, depth=depth, tokenizer=tokenizer
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                adapter_dir = compiler.compile_text(full_context, output_dir=tmp_dir, rank=rank)
                c_time = compiler.last_compile_time_ms or 0.0
                compile_times.append(c_time)

                # Inject adapter
                adapted_model = EngramCompiler.inject_engram(model, adapter_dir)

                # Evaluate retrieval
                if evaluator is not None:
                    success, output_text, e_time = evaluator(
                        adapted_model, tokenizer, question, expected_answer
                    )
                else:
                    success, output_text, e_time = evaluate_retrieval(
                        adapted_model, tokenizer, question, expected_answer
                    )

                eval_times.append(e_time)
                total_tests += 1
                if success:
                    passed_tests += 1

                results_matrix[c_key][d_key] = {
                    "success": success,
                    "target_tokens": c_len,
                    "actual_tokens": actual_tokens,
                    "depth": depth,
                    "compile_time_ms": c_time,
                    "eval_time_ms": e_time,
                    "output_preview": output_text[:100],
                }

    overall_accuracy = (passed_tests / total_tests) if total_tests > 0 else 0.0

    return {
        "benchmark": "Needle In A Haystack (NIAH)",
        "method": "EngramRS Zero-Copy Memory Compilation",
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "overall_accuracy": overall_accuracy,
        "mean_compile_time_ms": sum(compile_times) / len(compile_times) if compile_times else 0.0,
        "mean_eval_time_ms": sum(eval_times) / len(eval_times) if eval_times else 0.0,
        "context_lengths": parsed_lens,
        "depths": parsed_depths,
        "results": results_matrix,
    }


def format_niah_table(results: dict[str, Any]) -> str:
    """Format benchmark results into a clear ASCII matrix table."""
    matrix = results.get("results", {})
    if not matrix:
        return "No results to display."

    depth_keys = list(next(iter(matrix.values())).keys())

    # Header
    col_width = 12
    header_cols = ["Context"] + [f"Depth {d}" for d in depth_keys]
    header = " | ".join(col.center(col_width) for col in header_cols)
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "ENGRAM-RS NEEDLE IN A HAYSTACK BENCHMARK REPORT".center(len(header)),
        "=" * len(header),
        header,
        separator,
    ]

    for c_key, depths in matrix.items():
        row = [c_key.center(col_width)]
        for d_key in depth_keys:
            entry = depths.get(d_key, {})
            status = "PASS" if entry.get("success", False) else "FAIL"
            row.append(status.center(col_width))
        lines.append(" | ".join(row))

    lines.append(separator)
    lines.append(f"Total Tests: {results.get('total_tests', 0)}")
    lines.append(f"Passed: {results.get('passed_tests', 0)} ({results.get('overall_accuracy', 0.0) * 100:.1f}%)")
    lines.append(f"Mean Compilation Latency: {results.get('mean_compile_time_ms', 0.0):.2f} ms")
    lines.append(f"Mean Empty-Prompt Retrieval Latency: {results.get('mean_eval_time_ms', 0.0):.2f} ms")
    lines.append("=" * len(header))

    return "\n".join(lines)
