# EngramRS (v0.3.0)

**An Experimental Research Prototype for Zero-Grad KV-to-LoRA Analytical Compilation in Rust + Python.**  
EngramRS explores whether LLM context activations can be compiled into low-rank weight adapters (LoRA) in closed form ($\Delta W = Y X^+$) via Dual Woodbury Cholesky and Randomized SVD without backpropagation.

[![CI](https://github.com/qaddasd/EngramRS/actions/workflows/ci.yml/badge.svg)](https://github.com/qaddasd/EngramRS/actions/workflows/ci.yml)
[![Rust 1.81+](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Tests 104 passed](https://img.shields.io/badge/tests-104%20passed%20%7C%200%20failed-brightgreen.svg)]()
[![Clippy 0 warnings](https://img.shields.io/badge/clippy-0%20warnings-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary & Research Positioning

> [!WARNING]
> **Research Prototype Status & Scientific Integrity:**  
> EngramRS is an **experimental research prototype**, not a drop-in production replacement for KV-caching or RAG.  
> Our benchmarks on `Qwen/Qwen2.5-0.5B-Instruct` show that while analytical LoRA weight injection produces a measurable drop in target token perplexity (up to **-3.03%** on SwiGLU MLP layers), **Exact Fact Recall under greedy decoding without prompt context is 0/5 (0%)**.  
> The base model prior $W_{\text{base}}$ dominates during auto-regressive generation, and closed-form linear updates over 2000 tokens of running prose do not achieve the factual fidelity of full attention ($K, V$).

| Dimension | Standard Full Context | KV Prefix Caching (vLLM/SGLang) | EngramRS (LoRA $\Delta W$) |
|---|:---:|:---:|:---:|
| **Exact Fact Recall** | **100% (5/5)** | **100% (5/5)** | **0% (0/5)** (Prior dominates) |
| **Prefill Latency (TTFT)** | ~5500 ms (re-prefill) | **~60 ms** (cached) | **~60–95 ms** (prompt erased) |
| **Context KV Cache in VRAM** | 24.94 MiB | 24.94 MiB (held in RAM) | **0.19 MiB** (query-only KV) |
| **Adapter Storage Size** | 0 MB | 0 MB | 1.85 MiB (FP16) / 3.69 MiB (FP32) |
| **Setup Overhead** | 0 ms | 1x prefill | 400 ms (`o_proj`) – 7600 ms (Dual) |

---

## 🧪 Deep Benchmark Results (`Qwen/Qwen2.5-0.5B-Instruct`)

All metrics were captured on an **NVIDIA GeForce GTX 1650 (CUDA, float16)** using a clean base model per evaluation step with multi-run median synchronization.  
Benchmark script: [`benchmarks/run_complex_comparison.py`](benchmarks/run_complex_comparison.py), raw telemetry in [`benchmarks/complex_benchmark_results.json`](benchmarks/complex_benchmark_results.json).

### 1. 5-Domain Multi-Needle Haystack (~2000 Tokens)

Five distinct technical facts were embedded across fractional depths (15% to 95%) in 1993 tokens of background technical prose:

| Domain | Needle Depth | Full Context PPL | Zero Context PPL | Engram LoRA PPL | PPL Drop | TTFT Speedup | Exact Fact Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cryptography** | 15% | **1.461** | 85.855 | 85.665 | -0.22% | **92.2x** | 0/1 (0%) |
| **Distributed DBs** | 35% | **1.475** | 6.929 | 6.927 | -0.03% | **56.3x** | 0/1 (0%) |
| **Quantum Hardware** | 55% | **1.851** | 35.256 | 35.653 | +1.13% | **59.1x** | 0/1 (0%) |
| **Aerospace Eng.** | 75% | **1.710** | 13.485 | 13.474 | -0.08% | **57.6x** | 0/1 (0%) |
| **Biochemical Synth.**| 95% | **2.767** | 11.443 | **11.133** | **-2.72%** | **58.9x** | 0/1 (0%) |

#### What the model actually outputs:
- **Target Fact:** `Optimal thermal stabilization buffer pH for protein variant K-42 is 6.85.`
- **Full Context Output:** `"... buffer pH for protein variant K-42 is 6.85. This value was determined..."` (Correct, Recall 1/1)
- **Zero Context Output:** `"... buffer pH for protein variant K-42 was determined to be 7.5."` (Pretrained prior, Recall 0/1)
- **Engram LoRA Output:** `"... buffer pH for protein variant K-42 was determined to be 7.5."` (Pretrained prior, Recall 0/1)
- **Virtual Prefix Output:** `":[[:[[ORSспектListComponent:[[:[[\"),\":[[:[[:[[ORS..."` (RoPE phase collapse, Recall 0/1)

### 2. Context Length Scaling

| Context Length | Full KV Cache | Engram Query KV | Rust Compile Time | Full TTFT | Engram TTFT | TTFT Ratio | Prompt Compression |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **591 tokens** | 6.93 MiB | 0.19 MiB | 1486 ms | 1555 ms | 66 ms | 23.6x | 97.3% |
| **1156 tokens**| 13.55 MiB | 0.19 MiB | 3899 ms | 3174 ms | 231 ms | 13.7x | 98.6% |
| **2128 tokens**| 24.94 MiB | 0.19 MiB | 7675 ms | 5935 ms | 234 ms | 25.4x | 99.2% |

### 3. Sublayer Ablation (`o_proj` vs SwiGLU `down_proj`)

| Target Sublayer | Compile Time | Adapter Size (FP32) | Answer Perplexity | Relative PPL Shift |
|---|:---:|:---:|:---:|:---:|
| **Baseline (Clean Model, No Adapter)** | — | — | 85.855 | 0.00% |
| **Attention only (`o_proj`)** | **407 ms** | **898 KB** | 85.665 | -0.22% |
| **MLP SwiGLU only (`down_proj`)** | 6701 ms | 2882 KB | **83.251** | **-3.03%** |
| **Dual (`o_proj` + `down_proj`)** | 7138 ms | 3780 KB | 85.712 | -0.17% |

---

## 🔍 Scientific Analysis & Known Limitations

### 1. Why Exact Fact Recall is 0/5
1. **Prior Dominance ($W_{\text{base}} \gg \Delta W$):**  
   In standard PEFT LoRA, the forward pass is $h = x W_{\text{base}} + \frac{\alpha}{r} x A B$. For $r=16$ with $\alpha/r = 1.0$, the norm $\|\Delta W\|_F$ is small relative to $\|W_{\text{base}}\|_F$. While it shifts logit probabilities enough to lower perplexity by up to 3%, greedy $\text{argmax}$ decoding still selects the high-probability tokens from pretraining.
2. **Key Distribution Mismatch (Train vs Inference):**  
   During compilation, $\Delta W$ is fit on activations $X_{\text{ctx}}$ from reading the source document. At test time, the input is a short question prompt $X_{\text{query}}$. Because the query activations lie in a different subspace than the document activations, the linear projection does not fire strongly on query tokens unless guided by explicit activation steering.
3. **Contrast with ROME / MEMIT:**  
   Knowledge-editing methods like ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023) solve $W_{\text{new}} K = V$ for isolated, structured subject-relation-object keys. Fitting 2000 tokens of unstructured multi-topic prose into a single linear rank-16 map leads to substantial subspace interference.

### 2. Nature of the TTFT Latency Reduction
- The measured TTFT reduction (13.7x–92.2x) is entirely due to **prompt truncation** (processing 16 query tokens instead of 2000 context tokens), not faster per-token matrix arithmetic.
- Compiling the adapter takes 0.4–7.6 seconds. Therefore, compile-and-query is only amortizable if a context is queried repeatedly. If memory is available, **KV Prefix Caching** (vLLM / SGLang) is superior because it preserves 100% exact recall with identical ~60 ms TTFT.

### 3. RoPE Phase Degradation in Virtual KV-Prefix Tuning
- Decoders with Rotary Position Embeddings (RoPE) rely on continuous angular phase coordinates $\theta_t = t \cdot b^{-2i/d}$.
- Projecting 2000 tokens into 32 virtual prefix tokens breaks phase continuity, causing auto-regressive decoding loops. LoRA weight modification does not suffer from this issue and preserves linguistic coherence.

---

## 📚 Related Work

- **Locating and Editing Factual Associations (ROME / MEMIT):** Meng et al. (2022, 2023). Closed-form least-squares updates for factual knowledge editing in transformer MLP layers.
- **AlphaEdit:** Sun et al. (2024). Null-space projection for preserving background capabilities during knowledge editing.
- **Context Distillation & Prompt Tuning:** Snell et al. (2022), Lester et al. (2021). Distilling long prompts into soft tokens via iterative backpropagation.
- **Cartridges & Doc-to-LoRA:** Gheini et al. (2024). Training LoRA weights per document using gradient descent.
- **KV Cache Prefix Caching:** Kwon et al. (vLLM, 2023). Zero-information-loss caching of KV tensors in memory.

---

## 🛠️ Architecture & Implemented Audits

```
EngramRS/
├── .github/workflows/ci.yml    # Continuous Integration: fmt, clippy, cargo test, pytest
├── Cargo.toml                  # Rust 1.81+, faer, rayon, ahash, safetensors, half, pyo3
├── pyproject.toml              # Maturin configuration, CLI entry point [project.scripts] engram
├── src/                        # Rust Mathematical Kernel
│   ├── deduplication/          # fast_hash.rs: zero-alloc AHash, isotropic Box-Muller Gaussian SimHash
│   ├── math/                   # anchors.rs (rSVD), solvers.rs (Dual Woodbury Cholesky), factorize.rs
│   ├── rope_gqa.rs             # GQA head-mapping & RoPE transformations
│   ├── serializer/             # peft_config.rs (alpha_pattern, rank_pattern), safetensors.rs (FP16 & FP32)
│   ├── simd/                   # centering.rs (AVX2/AVX-512 sink-centering, BF16/FP16 unpack)
│   ├── streaming/              # accumulator.rs (Joint Frequent Directions [X | gamma Y])
│   └── lib.rs                  # PyO3 module interface
├── python/engram_rs/           # Python User-Facing API
│   ├── catchers/               # attention.py (o_proj basis alignment), mlp.py, base.py
│   ├── compiler.py             # EngramCompiler (safe unwrap, multi-target, layer_indices)
│   ├── cli.py                  # CLI utility `engram` (compile, merge, export-gguf, benchmark-niah)
│   ├── gguf.py                 # Pure-Python GGUF v3 exporter for llama.cpp & Ollama Modelfile
│   ├── merge.py                # Zero-Grad Model Merge with exact alpha/r scaling
│   └── null_space.py           # Strict Null-Space Orthogonal Projection
└── tests/                      # 104 passing tests (34 Rust tests + 70 Pytest tests)
```

### Audits & Fixes in v0.3.0
1. **`o_proj` Basis Realignment:** Hook directly attaches to `self_attn.o_proj` input, capturing the true multi-head concatenation rather than the pre-attention layernorm.
2. **Dimension-Aware Null-Space:** Strict null-space projections match the input dimension $d_{\text{in}}$ of the specific target module.
3. **PEFT Uniform Scaling (`alpha_pattern`):** Injected adapters specify $\alpha_i = r_i$ per layer, eliminating unwanted $4\times$ scaling variance across heterogeneous ranks.
4. **Joint Frequent Directions:** Replaced divergent $\frac{\sigma}{\tilde{\sigma}}$ heuristics with Frobenius-balanced joint sketches $[X \mid \gamma Y]$ where $\gamma = \sqrt{\frac{\|X\|_F^2 / d_{\text{in}}}{\|Y\|_F^2 / d_{\text{out}} + 10^{-12}}}$.
5. **Isotropic Gaussian SimHash:** Replaced LCG pseudo-random projections with standard Gaussian $\mathcal{N}(0, I)$ hyperplanes via the Box-Muller transform.
6. **PEFT State Hygiene:** `inject_engram` unloads prior adapters via `model.unload()` before re-wrapping to prevent layer accumulation.
7. **FP16 Storage Support:** Added `write_safetensors_typed` supporting 16-bit half precision, halving adapter file size.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/qaddasd/EngramRS.git
cd EngramRS

# Build and install the Rust extension in release mode
pip install maturin
maturin develop --release
```

### 1. Compile Context into LoRA (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_rs import EngramCompiler

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

compiler = EngramCompiler(model, tokenizer)
adapter_dir = compiler.compile_text(
    context_text=document_text,
    output_dir="./engram_adapter",
    targets=("o_proj", "down_proj"),
    layer_indices=range(8, 16),
    rank=16,
)

# Inject adapter into the model
model = EngramCompiler.inject_engram(model, adapter_dir)

# Query without the document in prompt
inputs = tokenizer("Question: What is the secret code?\nAnswer:", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Standalone Model Merge (Zero-Grad)

Fold $\Delta W$ directly into base model weights without PEFT dependencies:

```bash
engram merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter ./engram_adapter --output ./merged_model
```

### 3. GGUF Export for llama.cpp & Ollama

```bash
engram export-gguf --adapter ./engram_adapter --output ./adapter.gguf
```

- **For llama.cpp:**
  ```bash
  llama-cli -m model.gguf --lora adapter.gguf -p "Question: What is the secret code?\nAnswer:"
  ```
- **For Ollama:**
  Create a `Modelfile`:
  ```dockerfile
  FROM qwen2.5:0.5b
  ADAPTER ./adapter.gguf
  ```
  Then build and run:
  ```bash
  ollama create my-engram-model -f Modelfile
  ollama run my-engram-model
  ```

---

## 🧪 Running the Test Suite

```bash
# Rust Unit and Integration Tests (34 tests)
cargo test --verbose

# Rust Clippy (0 warnings)
cargo clippy --all-targets -- -D warnings

# Python Pytest Suite (70 tests)
pytest -v tests/
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
