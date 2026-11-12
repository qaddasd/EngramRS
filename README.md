# EngramRS (v0.3.0)

**A Research Prototype for a Three-Tier Long-Context Memory Foundation in Rust + Python.**
EngramRS explores how far LLM context activations can be compressed into persistent memory *without backpropagation*: an analytical low-rank weight tier ($\Delta W = Y X^+$ via Dual Woodbury Cholesky + Randomized SVD) for style/domain adaptation, and a **query-aware residual-KV tier** (SnapKV selection + RoPE re-anchoring) for verbatim fact recall after the source context is erased.

[![CI](https://github.com/qaddasd/EngramRS/actions/workflows/ci.yml/badge.svg)](https://github.com/qaddasd/EngramRS/actions/workflows/ci.yml)
[![Rust 1.81+](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Tests 54 Rust + 87 Python](https://img.shields.io/badge/tests-54%20Rust%20%2B%2087%20Python-brightgreen.svg)]()
[![Clippy 0 warnings](https://img.shields.io/badge/clippy-0%20warnings-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🇷🇺 Русская версия: [`README_RU.md`](README_RU.md) · Полный план переработки: [`docs/REDESIGN_V2.md`](docs/REDESIGN_V2.md)

---

## 📌 Executive Summary & Research Positioning

> [!WARNING]
> **Research prototype — read this before using.**
> EngramRS is an **experimental research prototype**, not a drop-in production replacement for KV-caching or RAG.
> Its central honest finding is that **verbatim fact recall and low-rank weight compression are different problems**:
> - **Pure-weight LoRA ($\Delta W$) cannot store arbitrary verbatim facts.** On a 5-needle haystack it scores **0/5 exact recall** — an information-theoretic limit, not a bug. It *does* shift answer perplexity (style/domain adaptation).
> - **A query-agnostic fixed KV-prefix is lossy**: it also scores **0/5** on buried multi-needle facts.
> - **A query-aware residual-KV prefix (SnapKV) recovers verbatim facts**: **100% on a single buried needle** after full context erasure (gated e2e test), and **2/5 exact + 3/5 near-miss** on the hard 5-needle benchmark at a tight 256-token budget.
>
> If you need guaranteed 100% recall on arbitrary multi-fact queries with memory to spare, **KV prefix caching (vLLM/SGLang) remains superior**. EngramRS targets the regime where the context must be *erased* and memory is constrained.

### The three-tier memory contract

| Tier | Mechanism | What it preserves | Verbatim fact recall | Cost |
|---|---|---|:---:|---|
| **1. Analytical LoRA** | $\Delta W = Y X^+$, rank-16, zero-grad | Style / domain perplexity | **0%** (information limit) | 0.4–7.9 s compile, 1.85 MiB FP16 |
| **2. Fixed KV-prefix** | Query-agnostic top-B selection | Lossy gist | **0/5** on buried needles | ~1 forward, 256-token prefix |
| **3. Query-aware residual-KV** | SnapKV (question = observation window) + RoPE re-anchor | **Exact fact tokens** | **100% single-needle** · **2/5 multi-needle @256** | ~1 forward/query, 256-token prefix |

| Dimension | Full Context | KV Prefix Cache (vLLM/SGLang) | EngramRS LoRA (Tier 1) | EngramRS Residual-KV (Tier 3) |
|---|:---:|:---:|:---:|:---:|
| **Exact recall — single needle** | 100% | 100% | **0%** | **100%** (validated e2e) |
| **Exact recall — 5-needle @256 budget** | 100% (5/5) | 100% (5/5) | **0% (0/5)** | **40% (2/5)** + 3 near-miss |
| **Prefill latency (TTFT)** | ~7000 ms | ~60 ms (cached) | ~55–110 ms (prompt erased) | ~61–108 ms (256-token prefix) |
| **Context KV in VRAM (2128 tok)** | 24.94 MiB | 24.94 MiB (held) | **0.19 MiB** (query-only) | **~3 MiB** (256-token prefix) |
| **Persistent artifact** | 0 | 0 | 1.85 MiB FP16 / 3.69 MiB FP32 | prefix KV (re-derived per query) |
| **Setup overhead** | 0 | 1× prefill | 0.4–7.9 s compile | ~1 forward pass per query |

---

## 🧪 Deep Benchmark Results (`Qwen/Qwen2.5-0.5B-Instruct`)

All metrics captured on an **NVIDIA GeForce GTX 1650 (CUDA, float16)**, clean base model per evaluation step, median of 2 warmup + 5 measured runs with CUDA synchronization.
Script: [`benchmarks/run_complex_comparison.py`](benchmarks/run_complex_comparison.py) · Raw telemetry: [`benchmarks/complex_benchmark_results.json`](benchmarks/complex_benchmark_results.json).

### 1. 5-Domain Multi-Needle Haystack (~2000 tokens)

Five technical facts embedded at fractional depths 15%–95% in 1993 tokens of background prose. The residual-KV prefix is compiled **per domain** with the question as the SnapKV observation window (256 tokens/domain, 7.8× compression).

| Domain | Depth | Full-ctx PPL | Base PPL | LoRA PPL | LoRA TTFT boost | LoRA recall | **Residual-KV recall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cryptography | 15% | **1.455** | 86.124 | 77.842 | 126.9× | 0/1 | 0/1 (near-miss) |
| Distributed DBs | 35% | **1.474** | 6.945 | 6.029 | 68.2× | 0/1 | 0/1 (near-miss) |
| Quantum Hardware | 55% | **1.852** | 35.181 | 25.887 | 68.4× | 0/1 | 0/1 (near-miss) |
| Aerospace Eng. | 75% | **1.714** | 13.602 | 17.622 | 65.6× | 0/1 | **1/1 ✓** |
| Biochemical Synth. | 95% | **2.768** | 11.500 | 9.629 | 70.5× | 0/1 | **1/1 ✓** |

**Exact fact recall summary:**

| Condition | Recall |
|---|:---:|
| Full Context (KV 1993t) | **5/5 (100%)** |
| Without Context (base prior) | 0/5 (0%) |
| Engram LoRA (Tier 1, weights only) | **0/5 (0%)** |
| Query-Aware Residual-KV (Tier 3, 256t) | **2/5 (40%)** + 3 near-miss |

LoRA lowers answer perplexity on 4/5 domains (up to **−26.4%** on Quantum) but *raises* it on Aerospace (+29.6%); exact verbatim recall stays 0/5 regardless. The residual-KV tier recovers 2 facts exactly and produces 3 **near-misses** that are digit-corruptions of complex identifiers (budget does not cover the full fact-span):

| Domain | Expected | Residual-KV output | Verdict |
|---|---|---|---|
| Aerospace | `3850 seconds` | `...is 3850 seconds.` | ✅ exact |
| Biochemical | `6.85` | `...K-42 is 6.85.` | ✅ exact |
| Cryptography | `CIPHER-7749-DELTA` | `CIPHER-7.7. DELTA` | ⚠️ near-miss |
| Distributed DBs | `172.28.94.215` | `172.20.215.215` | ⚠️ near-miss |
| Quantum Hardware | `142.6 microseconds` | `14.6 microseconds` | ⚠️ near-miss |

> **RoPE is fixed.** Earlier builds produced gibberish (`:[[:[[ORS...`) from virtual-prefix phase collapse. With pre-RoPE storage + re-anchoring (`src/rope_gqa.rs`), all residual-KV outputs above are **coherent language** — the remaining errors are fact-budget truncation, not phase corruption.

### 2. Context-Length Scaling (pure-weight erasure path)

Context fully erased into LoRA; only the ~16-token query remains. TTFT collapse comes from **prompt truncation**, not faster per-token math.

| Context | Full KV | Query-only KV | Rust compile | Full TTFT | Engram TTFT | Speedup | Prompt compression |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 591 tok | 6.93 MiB | 0.19 MiB | 1325 ms | 1965 ms | 55.5 ms | 35.4× | 97.3% |
| 1156 tok | 13.55 MiB | 0.19 MiB | 3933 ms | 4175 ms | 56.6 ms | 73.7× | 98.6% |
| 2128 tok | 24.94 MiB | 0.19 MiB | 7856 ms | 8170 ms | 58.2 ms | 140.3× | 99.2% |

### 3. Sublayer Ablation (`o_proj` vs SwiGLU `down_proj`)

| Target | Compile | Size (FP32) | Answer PPL | PPL drop vs base |
|---|:---:|:---:|:---:|:---:|
| Baseline (no adapter) | — | — | 86.124 | 0.00% |
| Attention only (`o_proj`) | **413 ms** | **898 KB** | 77.361 | −10.17% |
| MLP SwiGLU only (`down_proj`) | 6702 ms | 2882 KB | **75.634** | **−12.18%** |
| Dual (`o_proj` + `down_proj`) | 7017 ms | 3780 KB | 77.842 | −9.62% |

`down_proj` alone gives the largest perplexity drop here; combining targets is **not** monotonically better. None of these weight-only variants achieve verbatim recall (see §1).

### 4. Single-Needle Context-Erasure (gated e2e guard)

[`tests/test_e2e_erasure.py`](tests/test_e2e_erasure.py) buries one secret code at ~30% depth in 2898 tokens of Russian filler, erases the context, and asserts the query-aware residual-KV prefix recovers it verbatim:

| Metric | Value |
|---|---:|
| Context tokens (erased) | 2898 |
| Question tokens | 45 |
| Resident prefix tokens | 256 |
| Full KV cache | 33.96 MiB |
| Resident prefix KV | **3.00 MiB** |
| KV compression | **91.2%** |
| **Residual-KV fact recall** | **100%** (asserted) |
| Pure-LoRA fact recall (diagnostic) | 0% |

Actual greedy-decoding outputs (secret code `428719`, erased from the prompt):
- **Full-context ceiling:** `428719` ✓
- **Query-aware residual-KV:** `428719` ✓ (verbatim, context erased)
- **Pure-LoRA:** `12345678` ✗ (pretrained prior)

Run with `ENGRAM_RUN_E2E=1 pytest tests/test_e2e_erasure.py -m e2e -s`. A query-agnostic context-tail window **misses this needle (0%)** — only the question-as-observation-window retains it. This is the regression guard for the "no 0%" fact tier.

---

## 🔍 Scientific Analysis & Known Limitations

### 1. Why pure-weight LoRA recall is 0/5 (information-theoretic limit)
- **Prior dominance ($W_{\text{base}} \gg \Delta W$):** for rank $r=16$ with $\alpha/r=1.0$, $\|\Delta W\|_F$ is small relative to $\|W_{\text{base}}\|_F$. It shifts logits enough to move perplexity, but greedy $\argmax$ still selects high-probability pretrained tokens.
- **Key distribution mismatch:** $\Delta W$ is fit on document activations $X_{\text{ctx}}$; at inference the input is a short question $X_{\text{query}}$ in a different subspace, so the linear map does not fire on query tokens.
- **Capacity:** a single rank-16 linear map over 2000 tokens of multi-topic prose cannot losslessly store multiple arbitrary multi-token identifiers retrievable from a different subspace. Contrast ROME/MEMIT, which solve $W_{\text{new}}K=V$ for *isolated, structured* subject-relation-object keys. **Verbatim facts require the residual-KV tier, not weights alone.**

### 2. Why residual-KV multi-needle recall is 2/5, not 5/5 (budget sensitivity)
- SnapKV retains the top-B context tokens the question attends to. At B=256 over 2000 tokens, the budget covers short facts (`6.85`, `3850 seconds`) exactly but **truncates long identifiers** (`CIPHER-7749-DELTA`, IP addresses, `142.6`), corrupting digits.
- This is a **budget/fact-span coverage** tradeoff, not a fundamental failure: raising B to cover full fact-spans, or per-fact span preservation, improves recall. Single-needle recall is 100% at the same budget because the whole budget concentrates on one fact.
- Honest position: we do **not** claim 100% multi-needle recall at a tight budget. We claim a *working verbatim fact tier* (100% single-needle, 40% exact + 60% near-miss on hard multi-needle) where weights alone give 0%.

### 3. Nature of the TTFT reduction
- The 35×–140× TTFT reduction is entirely **prompt truncation** (16 query tokens, or 256 prefix tokens, instead of 2000 context tokens), not faster matrix arithmetic.
- Compilation takes 0.4–7.9 s. Compile-and-query amortizes only if a context is queried repeatedly. With memory available, **KV prefix caching** preserves 100% recall at the same ~60 ms TTFT.

### 4. RoPE phase handling (fixed)
- Decoders with Rotary Position Embeddings rely on continuous angular phase $\theta_t = t \cdot b^{-2i/d}$. Naively compressing 2000 tokens into a short prefix breaks phase continuity → decoding loops/gibberish.
- **Fix:** store keys pre-RoPE ($K_{\text{store}} = R(-p_{\text{orig}})K$), select (never merge) original tokens, and re-anchor on load ($K_{\text{load}} = R(p_{\text{new}})K_{\text{store}}$) to positions $0..B$. Outputs are now coherent (§1). Keys are *selected*, not averaged, because merging keys under RoPE is phase-incoherent; values may be compressed.

---

## 📚 Related Work

- **ROME / MEMIT** — Meng et al. (2022, 2023): closed-form least-squares factual editing in MLP layers for isolated subject-relation-object keys.
- **AlphaEdit** — Sun et al. (2024): null-space projection to preserve background capabilities during sequential editing.
- **SnapKV** — Li et al. (2024): query-aware observation-window KV selection (the basis of Tier 3).
- **DuoAttention / PyramidKV / KIVI / KVQuant** — retrieval-vs-streaming head separation, pyramidal budgets, KV quantization.
- **Context Distillation / Prompt Tuning** — Snell et al. (2022), Lester et al. (2021): soft-token distillation via backprop (EngramRS is zero-grad).
- **Cartridges / Doc-to-LoRA** — trained per-document KV/LoRA via gradient descent.
- **KV Prefix Caching** — Kwon et al. (vLLM, 2023): zero-loss KV reuse; the baseline EngramRS is compared against.

---

## 🛠️ Architecture & Implemented Audits

```
EngramRS/
├── .github/workflows/ci.yml    # CI: fmt, clippy -D warnings, cargo test, maturin --release, pytest -m "not e2e"
├── Cargo.toml                  # Rust 1.81+, faer, rayon, ahash, safetensors, half, pyo3
├── pyproject.toml              # Maturin config, CLI entry point [project.scripts] engram
├── src/                        # Rust mathematical kernel (core/critical path)
│   ├── deduplication/          # fast_hash.rs: zero-alloc AHash, isotropic Gaussian SimHash (Box-Muller)
│   ├── math/                   # anchors.rs (rSVD), solvers.rs (Dual Woodbury Cholesky), factorize.rs
│   ├── rope_gqa.rs             # GQA head-mapping + RoPE invert/re-anchor (Tier 3 phase fix)
│   ├── serializer/             # peft_config.rs (alpha_pattern, rank_pattern), safetensors.rs (FP16/FP32)
│   ├── simd/                   # centering.rs (AVX2/AVX-512 sink-centering, BF16/FP16 unpack)
│   ├── streaming/              # accumulator.rs (Joint Frequent Directions [X | gamma Y])
│   └── lib.rs                  # PyO3 module interface
├── python/engram_rs/           # Thin Python user-facing layer
│   ├── catchers/               # attention.py (o_proj basis alignment), mlp.py, base.py
│   ├── compiler.py             # EngramCompiler (multi-target, layer_indices, safe unwrap)
│   ├── prefix.py               # EngramPrefixTuner: query-aware SnapKV residual-KV (Tier 3)
│   ├── cli.py                  # CLI `engram` (compile, merge, export-gguf, compare-rag, benchmark-niah)
│   ├── gguf.py                 # Pure-Python GGUF v3 exporter (canonical blk.N.attn_output.lora_a)
│   ├── merge.py                # Zero-grad model merge with exact alpha/r scaling
│   ├── null_space.py           # Dimension-aware null-space orthogonal projection
│   └── benchmarks/             # rag_comparison.py (honest analytical cost model)
└── tests/                      # 54 Rust + 87 Python (86 CI + 1 gated GPU e2e)
```

### Audits & fixes in v0.3.0
1. **Query-aware SnapKV (Tier 3):** observation window = the question, not the context tail. Fixes the 0% buried-needle recall of query-agnostic prefixes.
2. **RoPE invert/re-anchor:** pre-RoPE key storage + phase re-anchoring on load (`src/rope_gqa.rs`); eliminates virtual-prefix gibberish.
3. **`o_proj` basis realignment:** hook attaches to `self_attn.o_proj` input (true multi-head concatenation), not pre-attention layernorm.
4. **Dimension-aware null-space:** projection matches each target's input dimension $d_{\text{in}}$; silent skips are configuration errors.
5. **PEFT uniform scaling (`alpha_pattern`):** $\alpha_i = r_i$ per layer, removing $4\times$ scaling variance across heterogeneous ranks.
6. **Joint Frequent Directions:** Frobenius-balanced joint sketch $[X \mid \gamma Y]$, $\gamma = \sqrt{\frac{\|X\|_F^2/d_{\text{in}}}{\|Y\|_F^2/d_{\text{out}} + 10^{-12}}}$.
7. **Isotropic Gaussian SimHash:** standard $\mathcal{N}(0,I)$ hyperplanes via Box-Muller (replaces LCG projections).
8. **PEFT state hygiene:** `inject_engram` unloads prior adapters before re-wrapping.
9. **FP16 storage:** `write_safetensors_typed` halves adapter size.
10. **Honest cost model:** `compare-rag` is an analytical top-k RAG vs resident-prefix model with a computed breakeven (no fabricated measurements).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/qaddasd/EngramRS.git
cd EngramRS
pip install maturin
maturin develop --release   # release is required: debug faer is ~100x slower
```

### 1. Query-aware residual-KV prefix (Tier 3 — verbatim facts)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_rs import EngramPrefixTuner

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", attn_implementation="eager"  # SnapKV needs eager attention
).eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

tuner = EngramPrefixTuner(model=model, tokenizer=tokenizer, prefix_len=256, preserve_sinks=1)

question = "What is the secret launch code?"
chat_question = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
)

# The question is the observation window: fact-bearing tokens are retained.
prefix = tuner.compile_prefix(
    context_text=document_text, prefix_len=256, selection="snapkv", query_text=chat_question
)

# Erase the document; generate from the short resident prefix + question only.
q_inputs = tokenizer(chat_question, return_tensors="pt")
gen_inputs = tuner.prepare_generate_inputs(q_inputs["input_ids"], prefix, attention_mask=q_inputs.get("attention_mask"))
out = model.generate(**gen_inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 2. Compile context into LoRA (Tier 1 — style/domain adaptation)

```python
from engram_rs import EngramCompiler

compiler = EngramCompiler(model, tokenizer)
adapter_dir = compiler.compile_text(
    context_text=document_text, output_dir="./engram_adapter",
    targets=("o_proj", "down_proj"), layer_indices=range(8, 16), rank=16,
)
model = EngramCompiler.inject_engram(model, adapter_dir)
# NOTE: weights alone do NOT recall verbatim facts (0%); use Tier 3 for facts.
```

### 3. Standalone model merge (zero-grad)

```bash
engram merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter ./engram_adapter --output ./merged_model
```

### 4. GGUF export for llama.cpp & Ollama

```bash
engram export-gguf --adapter ./engram_adapter --output ./adapter.gguf
```

- **llama.cpp:** `llama-cli -m model.gguf --lora adapter.gguf -p "Question: ...?\nAnswer:"`
- **Ollama:** `Modelfile` with `FROM qwen2.5:0.5b` + `ADAPTER ./adapter.gguf`, then `ollama create my-engram-model -f Modelfile && ollama run my-engram-model`

### 5. Analytical RAG cost comparison

```bash
engram compare-rag --context-tokens 10000 --rag-topk 2000 --resident-prefix 256 --compile-ms 1600
```

---

## 🧪 Running the Test Suite

```bash
cargo test --verbose                       # 54 Rust tests
cargo clippy --all-targets -- -D warnings  # 0 warnings
pytest -v tests/ -m "not e2e"              # 86 Python tests (CI gate)

# Gated GPU end-to-end erasure guard (downloads Qwen2.5-0.5B-Instruct):
ENGRAM_RUN_E2E=1 pytest tests/test_e2e_erasure.py -m e2e -s   # +1 e2e (100% single-needle recall)
```

---

## 📄 License

[MIT License](LICENSE).