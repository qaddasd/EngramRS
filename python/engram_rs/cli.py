"""EngramRS Command Line Interface (CLI).

Provides production commands for:
  - `engram compile`: Zero-copy activation capture and adapter compilation.
  - `engram merge`: Analytical zero-grad model merge into base weights.
  - `engram export-gguf`: Export LoRA adapter to llama.cpp/Ollama compatible GGUF format.
  - `engram benchmark-niah`: Needle In A Haystack benchmark runner.
  - `engram compare-rag`: RAG vs Engram cost, latency, and memory comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch


def read_input_text(input_path_str: str) -> str:
    """Read context text from a file or traverse all text files in a directory."""
    input_path = Path(input_path_str).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return input_path.read_text(encoding="utf-8", errors="replace")

    if input_path.is_dir():
        text_extensions = {".txt", ".md", ".json", ".rst", ".py", ".rs", ".c", ".cpp", ".h"}
        chunks: list[str] = []
        for file in sorted(input_path.rglob("*")):
            if file.is_file() and file.suffix.lower() in text_extensions:
                try:
                    content = file.read_text(encoding="utf-8", errors="replace").strip()
                    if content:
                        chunks.append(f"--- File: {file.name} ---\n{content}")
                except Exception:
                    continue
        if not chunks:
            raise ValueError(f"No readable text files found in directory: {input_path}")
        return "\n\n".join(chunks)

    raise ValueError(f"Invalid input path: {input_path}")


def cmd_compile(args: argparse.Namespace) -> int:
    """Handle `engram compile`."""
    print(f"[EngramRS] Loading model and tokenizer: {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engram_rs.compiler import EngramCompiler

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device_str == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device_str)

    context_text = read_input_text(args.input)
    print(f"[EngramRS] Loaded input context ({len(context_text)} characters).")

    compiler = EngramCompiler(model, tokenizer)
    raw_targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())

    if args.stream:
        print(f"[EngramRS] Streaming compilation (chunk_size={args.chunk_size}, rank={args.rank}) ...")
        out_dir = compiler.compile_stream(
            context_text,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            rank=args.rank,
            lambda_reg=args.lambda_reg,
        )
    else:
        print(f"[EngramRS] Standard zero-copy compilation (rank={args.rank}, targets={raw_targets}) ...")
        out_dir = compiler.compile_text(
            context_text,
            output_dir=args.output,
            rank=args.rank,
            targets=raw_targets,
            lambda_reg=args.lambda_reg,
            lambda_null=args.lambda_null,
        )

    compile_time = compiler.last_compile_time_ms or 0.0
    prompt_tokens = compiler.last_prompt_tokens or 0
    print(f"[EngramRS] Adapter compiled successfully!")
    print(f"  • Destination: {out_dir}")
    print(f"  • Processed Tokens: {prompt_tokens:,}")
    print(f"  • Rust Compilation Time: {compile_time:.2f} ms")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Handle `engram merge`."""
    print(f"[EngramRS] Merging adapter '{args.adapter}' into base model '{args.base}' ...")
    from engram_rs.merge import merge_adapter_to_base

    dtype = None
    if args.torch_dtype:
        dt_str = args.torch_dtype.lower()
        if dt_str in ("float16", "fp16"):
            dtype = torch.float16
        elif dt_str in ("bfloat16", "bf16"):
            dtype = torch.bfloat16
        elif dt_str in ("float32", "fp32"):
            dtype = torch.float32

    out_path = merge_adapter_to_base(
        base_model_name_or_path=args.base,
        adapter_path=args.adapter,
        output_dir=args.output,
        device=args.device,
        torch_dtype=dtype,
        safe_serialization=not args.no_safetensors,
    )

    print(f"[EngramRS] Zero-grad model merge complete!")
    print(f"  • Standalone HF Model: {out_path}")
    print(f"  • No PEFT dependency required for inference.")
    return 0


def cmd_export_gguf(args: argparse.Namespace) -> int:
    """Handle `engram export-gguf`."""
    print(f"[EngramRS] Exporting adapter '{args.adapter}' to GGUF LoRA format ...")
    from engram_rs.gguf import export_lora_to_gguf

    standard = not args.raw_names
    out_file = export_lora_to_gguf(
        adapter_dir=args.adapter,
        output_file=args.output,
        out_type=args.out_type,
        standard_names=standard,
        architecture=args.architecture,
    )

    size_mb = Path(out_file).stat().st_size / (1024 * 1024)
    print(f"[EngramRS] GGUF LoRA adapter successfully created!")
    print(f"  • Output file: {out_file} ({size_mb:.2f} MB)")
    print(f"  • Target architecture: {args.architecture}")
    print(f"  • Format: GGML_{args.out_type.upper()}")
    print(f"  • Usage: llama-cli -m model.gguf --lora {Path(out_file).name}")
    print(f"           ollama run model --lora {Path(out_file).name}")
    return 0


def cmd_benchmark_niah(args: argparse.Namespace) -> int:
    """Handle `engram benchmark-niah`."""
    print(f"[EngramRS] Running Needle In A Haystack (NIAH) benchmark on '{args.model}' ...")
    from engram_rs.benchmarks.niah import format_niah_table, run_niah_benchmark

    results = run_niah_benchmark(
        model=args.model,
        context_lens=args.context_lens,
        depths=args.depths,
        needle=args.needle,
        question=args.question,
        expected_answer=args.expected,
        rank=args.rank,
    )

    table = format_niah_table(results)
    print("\n" + table + "\n")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[EngramRS] Benchmark results saved to {out_path}")

    return 0


def cmd_compare_rag(args: argparse.Namespace) -> int:
    """Handle `engram compare-rag`."""
    from engram_rs.benchmarks.rag_comparison import (
        BenchmarkParams,
        compare_rag_vs_engram,
        format_rag_comparison_table,
    )

    q_counts = [int(x.strip()) for x in args.queries.split(",") if x.strip()]
    params = BenchmarkParams(
        context_tokens=args.context_len,
        query_tokens=args.query_len,
        answer_tokens=args.answer_len,
        input_price_per_m=args.input_price,
        output_price_per_m=args.output_price,
    )

    results = compare_rag_vs_engram(
        context_tokens=args.context_len,
        query_counts=q_counts,
        params=params,
    )

    table = format_rag_comparison_table(results)
    print("\n" + table + "\n")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[EngramRS] Comparison results saved to {out_path}")

    return 0


def cmd_merge_adapters(args: argparse.Namespace) -> int:
    """Handle `engram merge-adapters`."""
    print(f"[EngramRS] Combining {len(args.adapters)} LoRA adapters via SVD & orthogonalization ...")
    from engram_rs.merge import combine_lora_adapters

    out_path = combine_lora_adapters(
        adapter_paths=args.adapters,
        output_dir=args.output,
        target_rank=args.rank,
    )
    print(f"[EngramRS] Successfully created merged adapter at: {out_path} (rank={args.rank})")
    return 0


def cmd_unlearn(args: argparse.Namespace) -> int:
    """Handle `engram unlearn`."""
    import numpy as np
    from engram_rs.merge import unlearn_concept_from_adapter

    concept_path = Path(args.concept_file).expanduser().resolve()
    if not concept_path.is_file():
        raise FileNotFoundError(f"Concept file not found: {concept_path}")

    concept_mat = np.load(str(concept_path))
    print(f"[EngramRS] Scrubbing concept subspace (shape {concept_mat.shape}) from {args.adapter} ...")
    out_path = unlearn_concept_from_adapter(
        adapter_path=args.adapter,
        concept_activations=concept_mat,
        output_dir=args.output,
        tolerance=args.tolerance,
    )
    print(f"[EngramRS] Scrubbed adapter successfully written to: {out_path}")
    return 0


def cmd_profile_layers(args: argparse.Namespace) -> int:
    """Handle `engram profile-layers`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engram_rs.profiler import LayerSensitivityProfiler

    context = read_input_text(args.input)
    print(f"[EngramRS] Profiling causal layer sensitivity on model '{args.model}' ...")

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    scores = LayerSensitivityProfiler.profile(model, tokenizer, context, device=device)
    top_layers = LayerSensitivityProfiler.top_k_layers(scores, top_k=args.top_k)

    print("\n[Layer Sensitivity Scores ||Delta Y_l||_F / ||H_l||_F]")
    print("-" * 50)
    for l_idx in sorted(scores.keys()):
        marker = " [TOP]" if l_idx in top_layers else ""
        print(f"  Layer {l_idx:2d}: {scores[l_idx]:.6f}{marker}")
    print("-" * 50)
    print(f"Recommended top {args.top_k} layers for adaptation: {top_layers}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram",
        description="EngramRS: Zero-Copy Activation Capture, Model Merge, and GGUF Export CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. compile
    p_compile = subparsers.add_parser("compile", help="Compile context into an Engram adapter.")
    p_compile.add_argument("-m", "--model", required=True, help="Base model identifier or path.")
    p_compile.add_argument("-i", "--input", required=True, help="Text file or directory of text files.")
    p_compile.add_argument("-o", "--output", required=True, help="Output directory for adapter.")
    p_compile.add_argument("--targets", default="o_proj", help="Target modules (e.g. o_proj,down_proj).")
    p_compile.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank (default 16).")
    p_compile.add_argument("--stream", action="store_true", help="Enable streaming chunked compilation.")
    p_compile.add_argument("--chunk-size", type=int, default=2048, help="Tokens per chunk when streaming.")
    p_compile.add_argument("--device", default="auto", help="Device (cpu, cuda, auto).")
    p_compile.add_argument("--lambda-reg", type=float, default=1e-4, help="Ridge regularization parameter.")
    p_compile.add_argument("--lambda-null", type=float, default=1.0, help="Null state regularization weight.")
    p_compile.set_defaults(func=cmd_compile)

    # 2. merge
    p_merge = subparsers.add_parser("merge", help="Analytically zero-grad merge LoRA adapter into base model.")
    p_merge.add_argument("-b", "--base", required=True, help="Base model identifier or path.")
    p_merge.add_argument("-a", "--adapter", required=True, help="Path to adapter directory.")
    p_merge.add_argument("-o", "--output", required=True, help="Output directory for merged standalone model.")
    p_merge.add_argument("--device", default="cpu", help="Device to load base model on (default cpu).")
    p_merge.add_argument("--torch-dtype", default=None, help="Model dtype (e.g. float16, bfloat16, float32).")
    p_merge.add_argument("--no-safetensors", action="store_true", help="Disable safetensors serialization.")
    p_merge.set_defaults(func=cmd_merge)

    # 3. export-gguf
    p_gguf = subparsers.add_parser("export-gguf", help="Export PEFT safetensors LoRA into GGUF LoRA format.")
    p_gguf.add_argument("-a", "--adapter", required=True, help="Path to adapter directory.")
    p_gguf.add_argument("-o", "--output", required=True, help="Output .gguf file path.")
    p_gguf.add_argument("--out-type", default="f32", choices=["f32", "f16"], help="GGML weight dtype (f32 or f16).")
    p_gguf.add_argument("--raw-names", action="store_true", help="Keep raw base_model names instead of blk.N standard names.")
    p_gguf.add_argument("--architecture", default="llama", help="Target architecture (default llama).")
    p_gguf.set_defaults(func=cmd_export_gguf)

    # 4. benchmark-niah
    p_niah = subparsers.add_parser("benchmark-niah", help="Run Needle In A Haystack benchmark.")
    p_niah.add_argument("-m", "--model", required=True, help="Base model identifier or path.")
    p_niah.add_argument("--context-lens", default="1k,4k,8k", help="Context lengths (e.g. 1k,4k,8k).")
    p_niah.add_argument("--depths", default="0,25,50,75,100", help="Depths (e.g. 0,25,50,75,100).")
    p_niah.add_argument("--needle", default="The secret access passcode to the central vault is 92481-OMEGA.", help="Needle fact.")
    p_niah.add_argument("--question", default="What is the secret access passcode to the central vault?", help="Retrieval question.")
    p_niah.add_argument("--expected", default="92481-OMEGA", help="Expected substring in answer.")
    p_niah.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank (default 16).")
    p_niah.add_argument("-o", "--output", default=None, help="Optional output JSON path.")
    p_niah.set_defaults(func=cmd_benchmark_niah)

    # 5. compare-rag
    p_rag = subparsers.add_parser("compare-rag", help="Compute cost and latency comparison vs traditional RAG.")
    p_rag.add_argument("--context-len", type=int, default=10_000, help="Context length in tokens (default 10,000).")
    p_rag.add_argument("--query-len", type=int, default=50, help="Query length in tokens (default 50).")
    p_rag.add_argument("--answer-len", type=int, default=150, help="Answer length in tokens (default 150).")
    p_rag.add_argument("--input-price", type=float, default=2.50, help="Price per 1M input tokens ($).")
    p_rag.add_argument("--output-price", type=float, default=10.00, help="Price per 1M output tokens ($).")
    p_rag.add_argument("--queries", default="1,5,10,50,100,500,1000,5000,10000", help="Comma-separated query counts.")
    p_rag.add_argument("-o", "--output", default=None, help="Optional output JSON path.")
    p_rag.set_defaults(func=cmd_compare_rag)

    # 6. merge-adapters
    p_merge_ad = subparsers.add_parser("merge-adapters", help="Combine multiple LoRA adapters via SVD & orthogonalization.")
    p_merge_ad.add_argument("-a", "--adapters", nargs="+", required=True, help="List of adapter directory paths.")
    p_merge_ad.add_argument("-o", "--output", required=True, help="Output directory for merged adapter.")
    p_merge_ad.add_argument("-r", "--rank", type=int, default=16, help="Target rank for merged adapter.")
    p_merge_ad.set_defaults(func=cmd_merge_adapters)

    # 7. unlearn
    p_unlearn = subparsers.add_parser("unlearn", help="Unlearn concept subspace from LoRA adapter via orthogonal projection.")
    p_unlearn.add_argument("-a", "--adapter", required=True, help="Path to adapter directory.")
    p_unlearn.add_argument("-c", "--concept-file", required=True, help="Path to concept activations .npy file.")
    p_unlearn.add_argument("-o", "--output", required=True, help="Output directory for scrubbed adapter.")
    p_unlearn.add_argument("--tolerance", type=float, default=1e-5, help="Singular value cutoff tolerance.")
    p_unlearn.set_defaults(func=cmd_unlearn)

    # 8. profile-layers
    p_prof = subparsers.add_parser("profile-layers", help="Profile causal layer sensitivity scores ||Delta Y_l||_F / ||H_l||_F.")
    p_prof.add_argument("-m", "--model", required=True, help="Base model identifier or path.")
    p_prof.add_argument("-i", "--input", required=True, help="Text file or sample context.")
    p_prof.add_argument("--top-k", type=int, default=12, help="Number of top layers to report.")
    p_prof.add_argument("--device", default="auto", help="Device (cpu, cuda, auto).")
    p_prof.set_defaults(func=cmd_profile_layers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"[EngramRS Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
