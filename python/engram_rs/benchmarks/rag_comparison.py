"""Analytical RAG-vs-Engram cost / latency / memory model.

This is a *closed-form cost model*, not a measured benchmark. Every number is
derived from the explicit assumptions in `BenchmarkParams` (token prices,
prefill/decode throughput, model geometry). It exists to reason about breakeven
and asymptotic behaviour, not to claim a measured speedup.

Three serving regimes are modelled honestly:

  * Traditional RAG (top-k retrieval). Each query resends only the *retrieved*
    window (`rag_retrieved_tokens`), NOT the whole corpus. Modelling RAG as
    resending the entire corpus every query is a strawman; real retrievers
    return a small top-k window. Systems with prefix caching (vLLM / SGLang)
    can further amortise a *stable* retrieved prefix across queries — this
    model does not credit that, so it is an upper bound on RAG cost.

  * Engram soft/style tier (`resident_prefix_tokens = 0`). Context is distilled
    into LoRA weights and queries carry no resident context. This tier transfers
    style / domain priors but CANNOT reproduce verbatim facts: a rank-r weight
    delta cannot store arbitrary token strings (information-theoretic limit).

  * Engram fact tier (`resident_prefix_tokens > 0`, the default). Verbatim recall
    requires the residual-KV tier: a compressed, RoPE-correct KV prefix of
    `resident_prefix_tokens` is replayed each query. This is the regime that
    actually achieves needle recall, so its cost / latency is the default.

`engram_compile_ms` is a one-time, hardware/model-dependent compilation latency
(an assumption, not a universal constant). For reference, compiling
Qwen2.5-0.5B (o_proj + down_proj) measured ~1.6 s on a GTX 1650 in release
builds; debug builds are 100x+ slower. Set it to your own measurement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchmarkParams:
    context_tokens: int = 10_000          # full corpus, compiled once
    rag_retrieved_tokens: int = 2_000     # top-k window RAG resends per query
    resident_prefix_tokens: int = 256     # residual-KV prefix for verbatim facts (0 => soft/style tier)
    query_tokens: int = 50
    answer_tokens: int = 150
    input_price_per_m: float = 2.50       # $/1M input tokens (assumption)
    output_price_per_m: float = 10.00     # $/1M output tokens (assumption)
    prefill_tok_per_sec: float = 4000.0   # GPU prefill throughput (assumption)
    decode_tok_per_sec: float = 60.0      # GPU decode throughput (assumption)
    engram_compile_ms: float = 1600.0     # one-time compile latency (assumption; hardware/model dependent)
    model_layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128


def compute_kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int) -> int:
    """Compute fp16 KV cache memory in bytes for a given sequence length.

    Formula: 2 (K and V) * layers * kv_heads * head_dim * seq_len * 2 bytes (fp16)
    """
    return 2 * layers * kv_heads * head_dim * seq_len * 2


def compare_rag_vs_engram(
    context_tokens: int = 10_000,
    query_counts: list[int] | None = None,
    params: BenchmarkParams | None = None,
) -> dict[str, Any]:
    """Calculate cost, latency and memory metrics for RAG vs Engram (analytical)."""
    if params is None:
        params = BenchmarkParams(context_tokens=context_tokens)
    else:
        params.context_tokens = context_tokens

    if query_counts is None:
        query_counts = [1, 5, 10, 50, 100, 500, 1000, 5000, 10_000]

    # A retriever cannot return more than the corpus contains; the resident
    # prefix is a compression of the corpus, so it cannot exceed it either.
    rag_retrieved = min(params.rag_retrieved_tokens, params.context_tokens)
    resident_prefix = min(params.resident_prefix_tokens, params.context_tokens)

    # --- Cost Analysis ---
    # RAG resends the retrieved top-k window + query every time.
    rag_input_tokens_per_query = rag_retrieved + params.query_tokens
    rag_cost_per_query = (
        (rag_input_tokens_per_query * params.input_price_per_m / 1_000_000.0)
        + (params.answer_tokens * params.output_price_per_m / 1_000_000.0)
    )

    # Engram compiles the full corpus once (one-time prefill cost).
    engram_compilation_cost = params.context_tokens * params.input_price_per_m / 1_000_000.0

    # Each Engram query replays the resident prefix (fact tier) + query.
    # resident_prefix == 0 models the soft/style tier (no verbatim recall).
    engram_input_tokens_per_query = resident_prefix + params.query_tokens
    engram_cost_per_query = (
        (engram_input_tokens_per_query * params.input_price_per_m / 1_000_000.0)
        + (params.answer_tokens * params.output_price_per_m / 1_000_000.0)
    )

    # Breakeven: Q * rag_cost = compile_cost + Q * engram_cost
    cost_delta_per_query = rag_cost_per_query - engram_cost_per_query
    breakeven_queries = (
        (engram_compilation_cost / cost_delta_per_query)
        if cost_delta_per_query > 0
        else math.inf
    )

    # --- Latency Analysis (TTFT) ---
    rag_ttft_ms = (rag_input_tokens_per_query / params.prefill_tok_per_sec) * 1000.0
    engram_ttft_ms = (engram_input_tokens_per_query / params.prefill_tok_per_sec) * 1000.0
    ttft_speedup = (rag_ttft_ms / engram_ttft_ms) if engram_ttft_ms > 0 else 1.0

    decode_time_ms = (params.answer_tokens / params.decode_tok_per_sec) * 1000.0
    rag_total_query_lat_ms = rag_ttft_ms + decode_time_ms
    engram_total_query_lat_ms = engram_ttft_ms + decode_time_ms
    # Compilation is a one-time setup cost paid before the first query.
    engram_first_query_lat_ms = params.engram_compile_ms + engram_total_query_lat_ms

    # --- Memory Analysis (KV cache footprint per session) ---
    rag_kv_bytes = compute_kv_cache_bytes(
        params.model_layers, params.kv_heads, params.head_dim, rag_input_tokens_per_query
    )
    engram_kv_bytes = compute_kv_cache_bytes(
        params.model_layers, params.kv_heads, params.head_dim, engram_input_tokens_per_query
    )
    kv_reduction_factor = rag_kv_bytes / engram_kv_bytes if engram_kv_bytes > 0 else 1.0

    curve_data = []
    for q in query_counts:
        rag_tot_cost = q * rag_cost_per_query
        engram_tot_cost = engram_compilation_cost + (q * engram_cost_per_query)
        savings_pct = (
            ((rag_tot_cost - engram_tot_cost) / rag_tot_cost * 100.0)
            if rag_tot_cost > 0
            else 0.0
        )
        curve_data.append({
            "queries": q,
            "rag_total_cost_usd": round(rag_tot_cost, 4),
            "engram_total_cost_usd": round(engram_tot_cost, 4),
            "cost_savings_usd": round(rag_tot_cost - engram_tot_cost, 4),
            "cost_savings_percent": round(savings_pct, 2),
        })

    regime = "fact (residual-KV prefix)" if resident_prefix > 0 else "soft/style (weights only, no verbatim recall)"

    return {
        "model_kind": "analytical cost model (assumptions, not measurements)",
        "regime": regime,
        "parameters": asdict(params),
        "effective_tokens": {
            "rag_retrieved_tokens": rag_retrieved,
            "resident_prefix_tokens": resident_prefix,
            "rag_input_tokens_per_query": rag_input_tokens_per_query,
            "engram_input_tokens_per_query": engram_input_tokens_per_query,
        },
        "breakeven_queries": (round(breakeven_queries, 2) if math.isfinite(breakeven_queries) else None),
        "one_time_compilation": {
            "cost_usd": round(engram_compilation_cost, 6),
            "latency_ms": round(params.engram_compile_ms, 2),
            "engram_first_query_latency_ms": round(engram_first_query_lat_ms, 2),
        },
        "per_query_comparison": {
            "rag_cost_usd": round(rag_cost_per_query, 6),
            "engram_cost_usd": round(engram_cost_per_query, 6),
            "rag_ttft_ms": round(rag_ttft_ms, 2),
            "engram_ttft_ms": round(engram_ttft_ms, 2),
            "ttft_speedup_x": round(ttft_speedup, 1),
            "rag_total_latency_ms": round(rag_total_query_lat_ms, 2),
            "engram_total_latency_ms": round(engram_total_query_lat_ms, 2),
        },
        "memory_comparison": {
            "rag_kv_cache_mb": round(rag_kv_bytes / (1024 * 1024), 2),
            "engram_kv_cache_mb": round(engram_kv_bytes / (1024 * 1024), 2),
            "kv_cache_reduction_x": round(kv_reduction_factor, 1),
        },
        "query_cost_curves": curve_data,
    }


def format_rag_comparison_table(results: dict[str, Any]) -> str:
    """Format the analytical RAG-vs-Engram model into an ASCII table."""
    params = results["parameters"]
    eff = results["effective_tokens"]
    pq = results["per_query_comparison"]
    mem = results["memory_comparison"]
    one = results["one_time_compilation"]
    curves = results["query_cost_curves"]

    breakeven = results["breakeven_queries"]
    if breakeven is None:
        breakeven_str = "never (Engram per-query cost >= RAG)"
    else:
        first_cheaper = max(1, math.floor(breakeven) + 1)
        breakeven_str = f"{breakeven} queries (Engram cheaper from query {first_cheaper} onward)"

    lines = [
        "=" * 84,
        "ENGRAM-RS VS TRADITIONAL RAG: ANALYTICAL COST MODEL".center(84),
        f"(Corpus: {params['context_tokens']:,}t | RAG top-k: {eff['rag_retrieved_tokens']:,}t | "
        f"Resident prefix: {eff['resident_prefix_tokens']:,}t)".center(84),
        f"Regime: {results['regime']}".center(84),
        "=" * 84,
        "",
        "--- PER-QUERY METRICS (steady state, after one-time compilation) ---",
        f"  TTFT:            RAG = {pq['rag_ttft_ms']} ms  vs  Engram = {pq['engram_ttft_ms']} ms  ({pq['ttft_speedup_x']}x)",
        f"  KV / session:    RAG = {mem['rag_kv_cache_mb']} MB  vs  Engram = {mem['engram_kv_cache_mb']} MB  ({mem['kv_cache_reduction_x']}x)",
        f"  Cost / query:    RAG = ${pq['rag_cost_usd']:.5f}  vs  Engram = ${pq['engram_cost_usd']:.5f}",
        f"  Breakeven:       {breakeven_str}",
        "",
        "--- ONE-TIME COMPILATION (paid once, before first query) ---",
        f"  Cost: ${one['cost_usd']:.5f}   Latency: {one['latency_ms']} ms   "
        f"First-query end-to-end: {one['engram_first_query_latency_ms']} ms",
        "",
        "--- CUMULATIVE COST CURVE ($ USD) ---",
        f"{'Queries':>10} | {'Traditional RAG':>18} | {'EngramRS':>18} | {'Cost Savings':>16} | {'Savings %':>11}",
        "-" * 84,
    ]

    for pt in curves:
        q_str = f"{pt['queries']:,}"
        rag_str = f"${pt['rag_total_cost_usd']:,.2f}"
        engram_str = f"${pt['engram_total_cost_usd']:,.2f}"
        saved_str = f"${pt['cost_savings_usd']:,.2f}"
        pct_str = f"{pt['cost_savings_percent']:.1f}%"
        lines.append(f"{q_str:>10} | {rag_str:>18} | {engram_str:>18} | {saved_str:>16} | {pct_str:>11}")

    lines += [
        "=" * 84,
        "Assumptions, not measurements. RAG resends only its top-k window (prefix",
        "caching would lower RAG cost further). Verbatim fact recall requires the",
        "residual-KV prefix; resident_prefix_tokens=0 models the style-only tier.",
        "=" * 84,
    ]
    return "\n".join(lines)
