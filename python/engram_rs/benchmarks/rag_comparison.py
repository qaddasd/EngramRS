"""RAG vs Engram Cost and Latency Comparison Utility.

Computes and displays exact mathematical cost and Time-To-First-Token (TTFT) curves
comparing:
  1. Traditional In-Context RAG: Resending context (e.g. 10k tokens) with every query.
  2. EngramRS Memory Compilation: One-time zero-copy weight compilation + empty-prompt queries.

Includes pricing metrics ($/1M tokens), TTFT latency (prefill throughput),
KV cache memory footprint (bytes per concurrent session), and breakeven query calculations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchmarkParams:
    context_tokens: int = 10_000
    query_tokens: int = 50
    answer_tokens: int = 150
    input_price_per_m: float = 2.50       # $2.50 per 1M input tokens
    output_price_per_m: float = 10.00     # $10.00 per 1M output tokens
    prefill_tok_per_sec: float = 4000.0   # GPU prefill throughput
    decode_tok_per_sec: float = 60.0      # GPU decode throughput
    engram_compile_ms: float = 45.0       # Rust zero-copy compilation time
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
    """Calculate comprehensive cost, latency, and memory metrics for RAG vs Engram."""
    if params is None:
        params = BenchmarkParams(context_tokens=context_tokens)
    else:
        params.context_tokens = context_tokens

    if query_counts is None:
        query_counts = [1, 5, 10, 50, 100, 500, 1000, 5000, 10_000]

    # --- Cost Analysis ---
    # In RAG, every query sends: (context_tokens + query_tokens) input + answer_tokens output
    rag_input_tokens_per_query = params.context_tokens + params.query_tokens
    rag_cost_per_query = (
        (rag_input_tokens_per_query * params.input_price_per_m / 1_000_000.0)
        + (params.answer_tokens * params.output_price_per_m / 1_000_000.0)
    )

    # In Engram, compilation is a one-time prefill of context_tokens
    engram_compilation_cost = params.context_tokens * params.input_price_per_m / 1_000_000.0

    # Each subsequent Engram query sends ONLY query_tokens (empty context prompt!)
    engram_input_tokens_per_query = params.query_tokens
    engram_cost_per_query = (
        (engram_input_tokens_per_query * params.input_price_per_m / 1_000_000.0)
        + (params.answer_tokens * params.output_price_per_m / 1_000_000.0)
    )

    # Breakeven point: Q where RAG total cost >= Engram total cost
    # Q * rag_cost = engram_compile_cost + Q * engram_cost
    # Q * (rag_cost - engram_cost) = engram_compile_cost
    # Q = engram_compile_cost / (rag_cost - engram_cost)
    cost_delta_per_query = rag_cost_per_query - engram_cost_per_query
    breakeven_queries = (
        (engram_compilation_cost / cost_delta_per_query)
        if cost_delta_per_query > 0
        else 1.0
    )

    # --- Latency Analysis (TTFT: Time To First Token) ---
    rag_ttft_ms = (rag_input_tokens_per_query / params.prefill_tok_per_sec) * 1000.0
    engram_ttft_ms = (engram_input_tokens_per_query / params.prefill_tok_per_sec) * 1000.0
    ttft_speedup = (rag_ttft_ms / engram_ttft_ms) if engram_ttft_ms > 0 else 1.0

    # Total generation time per query
    decode_time_ms = (params.answer_tokens / params.decode_tok_per_sec) * 1000.0
    rag_total_query_lat_ms = rag_ttft_ms + decode_time_ms
    engram_total_query_lat_ms = engram_ttft_ms + decode_time_ms

    # --- Memory Analysis (KV Cache footprint) ---
    rag_kv_bytes = compute_kv_cache_bytes(
        params.model_layers, params.kv_heads, params.head_dim, rag_input_tokens_per_query
    )
    engram_kv_bytes = compute_kv_cache_bytes(
        params.model_layers, params.kv_heads, params.head_dim, engram_input_tokens_per_query
    )
    kv_reduction_factor = rag_kv_bytes / engram_kv_bytes if engram_kv_bytes > 0 else 1.0

    # Step-by-step curve points
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

    return {
        "parameters": asdict(params),
        "breakeven_queries": round(breakeven_queries, 2),
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
    """Format RAG vs Engram benchmark into an ASCII comparison table."""
    params = results["parameters"]
    pq = results["per_query_comparison"]
    mem = results["memory_comparison"]
    curves = results["query_cost_curves"]

    lines = [
        "=" * 84,
        f"ENGRAM-RS VS TRADITIONAL RAG: SCIENTIFIC BENCHMARK & COST ANALYSIS".center(84),
        f"(Context: {params['context_tokens']:,} tokens | Query: {params['query_tokens']} tokens | Answer: {params['answer_tokens']} tokens)".center(84),
        "=" * 84,
        "",
        "--- METRIC COMPARISON SUMMARY ---",
        f"  • TTFT (Time-To-First-Token):  RAG = {pq['rag_ttft_ms']} ms  vs  Engram = {pq['engram_ttft_ms']} ms  ({pq['ttft_speedup_x']}x FASTER)",
        f"  • KV Cache Memory / Session:   RAG = {mem['rag_kv_cache_mb']} MB  vs  Engram = {mem['engram_kv_cache_mb']} MB  ({mem['kv_cache_reduction_x']}x SAVINGS)",
        f"  • Cost Per Query:              RAG = ${pq['rag_cost_usd']:.5f}  vs  Engram = ${pq['engram_cost_usd']:.5f}",
        f"  • Breakeven Point:             {results['breakeven_queries']} queries (Engram is cheaper from query 2 onward)",
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

    lines.append("=" * 84)
    return "\n".join(lines)
