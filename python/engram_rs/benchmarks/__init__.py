"""Scientific Benchmarks and Validation for EngramRS."""

from .niah import run_niah_benchmark, format_niah_table
from .rag_comparison import compare_rag_vs_engram, format_rag_comparison_table

__all__ = [
    "run_niah_benchmark",
    "format_niah_table",
    "compare_rag_vs_engram",
    "format_rag_comparison_table",
]
