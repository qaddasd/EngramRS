"""EngramRS activation capture, streaming covariance compilation, null-space projection, and ecosystem tools."""

from .prefix import (
    EngramPrefixTuner,
    rope_apply_kv,
    rope_inverse_kv,
    select_compress_kv,
    snapkv_importance,
    spectral_compress_kv,
)
from .gpu_solver import (
    GPULayerCompiler,
    build_peft_state_dict,
    solve_all_layers_gpu,
    solve_layer_gpu,
)
from .hooks import EngramContextCatcher
from .merge import (
    combine_lora_adapters,
    compute_delta_w,
    merge_adapter_to_base,
    merge_lora_into_model,
    unlearn_concept_from_adapter,
)
from .profiler import LayerSensitivityProfiler
from .gguf import (
    GGUFReader,
    GGUFWriter,
    export_lora_to_gguf,
    peft_to_gguf_tensor_name,
)
from .null_space import (
    DEFAULT_CALIBRATION_PROMPTS,
    calibrate_background_anchors,
    compute_null_space_projector,
    get_family_background_anchors,
    project_lora_to_null_space,
    project_to_null_space,
)
from .randomized_svd import randomized_svd
from .streaming import StreamingCovarianceAccumulator
from .benchmarks import (
    compare_rag_vs_engram,
    format_niah_table,
    format_rag_comparison_table,
    run_niah_benchmark,
)

__all__ = [
    "DEFAULT_CALIBRATION_PROMPTS",
    "EngramCompiler",
    "EngramContextCatcher",
    "EngramPrefixTuner",
    "GGUFReader",
    "GGUFWriter",
    "GPULayerCompiler",
    "LayerSensitivityProfiler",
    "StreamingCovarianceAccumulator",
    "build_peft_state_dict",
    "calibrate_background_anchors",
    "combine_lora_adapters",
    "compare_rag_vs_engram",
    "compile_all_layers_to_safetensors",
    "compute_delta_w",
    "compute_null_space_projector",
    "export_lora_to_gguf",
    "format_niah_table",
    "format_rag_comparison_table",
    "get_family_background_anchors",
    "merge_adapter_to_base",
    "merge_lora_into_model",
    "peft_to_gguf_tensor_name",
    "project_lora_to_null_space",
    "project_to_null_space",
    "randomized_svd",
    "rope_apply_kv",
    "rope_inverse_kv",
    "run_niah_benchmark",
    "select_compress_kv",
    "snapkv_importance",
    "solve_all_layers_gpu",
    "solve_layer_gpu",
    "spectral_compress_kv",
    "unlearn_concept_from_adapter",
]


def __getattr__(name: str):
    if name == "EngramCompiler":
        from .compiler import EngramCompiler

        globals()[name] = EngramCompiler
        return EngramCompiler
    if name == "compile_all_layers_to_safetensors":
        from .engram_rs import compile_all_layers_to_safetensors

        globals()[name] = compile_all_layers_to_safetensors
        return compile_all_layers_to_safetensors
    if name in ("center_and_unpack_bf16_py", "center_f32_py", "RustStreamingAccumulator"):
        from . import engram_rs
        val = getattr(engram_rs, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
