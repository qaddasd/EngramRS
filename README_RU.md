# EngramRS (v0.3.0) — Документация на русском языке

**Исследовательский прототип трёхуровневой фундаментальной памяти для длинного контекста на Rust + Python.**
EngramRS исследует, насколько далеко можно сжать активации контекста LLM в persistent-память **без градиентного спуска**: аналитический низкоранговый весовой уровень ($\Delta W = Y X^+$ через Dual Woodbury Cholesky + рандомизированный SVD) для стиля/домена и **query-aware residual-KV уровень** (SnapKV-селекция + RoPE re-anchoring) для дословного воспроизведения фактов после стирания исходного контекста.

[![CI](https://github.com/qaddasd/EngramRS/actions/workflows/ci.yml/badge.svg)](https://github.com/qaddasd/EngramRS/actions/workflows/ci.yml)
[![Rust 1.81+](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Tests 54 Rust + 87 Python](https://img.shields.io/badge/tests-54%20Rust%20%2B%2087%20Python-brightgreen.svg)]()
[![Clippy 0 warnings](https://img.shields.io/badge/clippy-0%20warnings-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🇬🇧 English version: [`README.md`](README.md) · Полный план переработки: [`docs/REDESIGN_V2.md`](docs/REDESIGN_V2.md)

---

## 📌 Статус проекта и научная честность

> [!WARNING]
> **Исследовательский прототип — прочитайте перед использованием.**
> EngramRS — **экспериментальный прототип**, а не готовая промышленная замена KV-кэшированию или RAG.
> Центральный честный вывод: **дословное воспроизведение фактов и низкоранговое сжатие весов — разные задачи**:
> - **Чистые веса LoRA ($\Delta W$) не хранят произвольные дословные факты.** На 5-needle haystack — **0/5 exact recall**. Это информационно-теоретический предел, а не баг. Веса *смещают* перплексию ответа (стиль/домен).
> - **Query-agnostic фиксированный KV-префикс теряет информацию**: тоже **0/5** на закопанных multi-needle фактах.
> - **Query-aware residual-KV префикс (SnapKV) восстанавливает факты дословно**: **100% на одиночной закопанной игле** после полного стирания контекста (gated e2e-тест) и **2/5 exact + 3/5 near-miss** на сложном 5-needle бенчмарке при жёстком бюджете 256 токенов.
>
> Если нужна гарантированная 100% точность на произвольных multi-fact запросах и есть память — **KV prefix caching (vLLM/SGLang) остаётся лучше**. EngramRS целится в режим, где контекст должен быть *стёрт*, а память ограничена.

### Контракт трёхуровневой памяти

| Уровень | Механизм | Что сохраняет | Дословный recall | Цена |
|---|---|---|:---:|---|
| **1. Аналитический LoRA** | $\Delta W = Y X^+$, rank-16, zero-grad | Стиль / доменная перплексия | **0%** (инф. предел) | 0.4–7.9 с компиляция, 1.85 MiB FP16 |
| **2. Фикс. KV-префикс** | Query-agnostic top-B селекция | Lossy суть | **0/5** на закопанных иглах | ~1 forward, префикс 256 ток. |
| **3. Query-aware residual-KV** | SnapKV (окно = запрос) + RoPE re-anchor | **Точные токены факта** | **100% single-needle** · **2/5 multi-needle @256** | ~1 forward/запрос, префикс 256 ток. |

| Измерение | Полный контекст | KV Prefix Cache (vLLM/SGLang) | EngramRS LoRA (Ур. 1) | EngramRS Residual-KV (Ур. 3) |
|---|:---:|:---:|:---:|:---:|
| **Exact recall — одиночная игла** | 100% | 100% | **0%** | **100%** (валидировано e2e) |
| **Exact recall — 5-needle @256** | 100% (5/5) | 100% (5/5) | **0% (0/5)** | **40% (2/5)** + 3 near-miss |
| **Задержка Prefill (TTFT)** | ~7000 мс | ~60 мс (из кэша) | ~55–110 мс (промпт стёрт) | ~61–108 мс (префикс 256 ток.) |
| **KV контекста в VRAM (2128 ток.)** | 24.94 MiB | 24.94 MiB (удерж.) | **0.19 MiB** (только запрос) | **~3 MiB** (префикс 256 ток.) |
| **Persistent-артефакт** | 0 | 0 | 1.85 MiB FP16 / 3.69 MiB FP32 | prefix KV (пересоздаётся на запрос) |
| **Оверхед** | 0 | 1× prefill | 0.4–7.9 с компиляция | ~1 forward на запрос |

---

## 🧪 Детальные бенчмарки (`Qwen/Qwen2.5-0.5B-Instruct`)

Замеры на **NVIDIA GeForce GTX 1650 (CUDA, float16)**, чистая базовая модель на каждом этапе, медиана 2 warmup + 5 замеров с CUDA-синхронизацией.
Скрипт: [`benchmarks/run_complex_comparison.py`](benchmarks/run_complex_comparison.py) · Сырые данные: [`benchmarks/complex_benchmark_results.json`](benchmarks/complex_benchmark_results.json).

### 1. 5-доменный Multi-Needle Haystack (~2000 токенов)

5 технических фактов на глубинах 15%–95% в 1993 токенах фоновой прозы. Residual-KV префикс компилируется **на каждый домен** с запросом в качестве observation-window SnapKV (256 токенов/домен, сжатие 7.8×).

| Домен | Глубина | PPL с контекстом | PPL базы | PPL LoRA | TTFT boost LoRA | Recall LoRA | **Recall Residual-KV** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Криптография | 15% | **1.455** | 86.124 | 77.842 | 126.9× | 0/1 | 0/1 (near-miss) |
| Распределённые БД | 35% | **1.474** | 6.945 | 6.029 | 68.2× | 0/1 | 0/1 (near-miss) |
| Квантовое железо | 55% | **1.852** | 35.181 | 25.887 | 68.4× | 0/1 | 0/1 (near-miss) |
| Аэрокосмическая инж. | 75% | **1.714** | 13.602 | 17.622 | 65.6× | 0/1 | **1/1 ✓** |
| Биохимический синтез | 95% | **2.768** | 11.500 | 9.629 | 70.5× | 0/1 | **1/1 ✓** |

**Сводка exact fact recall:**

| Условие | Recall |
|---|:---:|
| Полный контекст (KV 1993t) | **5/5 (100%)** |
| Без контекста (базовый prior) | 0/5 (0%) |
| Engram LoRA (Ур. 1, только веса) | **0/5 (0%)** |
| Query-Aware Residual-KV (Ур. 3, 256t) | **2/5 (40%)** + 3 near-miss |

LoRA снижает перплексию ответа на 4/5 доменов (до **−26.4%** на Quantum), но *повышает* на Aerospace (+29.6%); дословный recall остаётся 0/5 в любом случае. Residual-KV уровень восстанавливает 2 факта точно и даёт 3 **near-miss** — искажения цифр сложных идентификаторов (бюджет не покрывает весь fact-span):

| Домен | Ожидается | Вывод Residual-KV | Вердикт |
|---|---|---|---|
| Аэрокосмическая | `3850 seconds` | `...is 3850 seconds.` | ✅ точно |
| Биохимическая | `6.85` | `...K-42 is 6.85.` | ✅ точно |
| Криптография | `CIPHER-7749-DELTA` | `CIPHER-7.7. DELTA` | ⚠️ near-miss |
| Распределённые БД | `172.28.94.215` | `172.20.215.215` | ⚠️ near-miss |
| Квантовое железо | `142.6 microseconds` | `14.6 microseconds` | ⚠️ near-miss |

> **RoPE исправлен.** Ранние сборки выдавали gibberish (`:[[:[[ORS...`) из-за фазового коллапса виртуального префикса. С pre-RoPE хранением + re-anchoring (`src/rope_gqa.rs`) все выводы residual-KV выше — **связный язык**; оставшиеся ошибки — усечение fact-бюджета, а не фазовая порча.

### 2. Масштабирование по длине контекста (путь стирания в веса)

Контекст полностью стёрт в LoRA; остаётся только запрос ~16 токенов. Коллапс TTFT — из-за **усечения промпта**, не из-за более быстрой арифметики.

| Контекст | Полный KV | KV запроса | Компиляция Rust | Полный TTFT | Engram TTFT | Ускорение | Сжатие промпта |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 591 ток. | 6.93 MiB | 0.19 MiB | 1325 мс | 1965 мс | 55.5 мс | 35.4× | 97.3% |
| 1156 ток. | 13.55 MiB | 0.19 MiB | 3933 мс | 4175 мс | 56.6 мс | 73.7× | 98.6% |
| 2128 ток. | 24.94 MiB | 0.19 MiB | 7856 мс | 8170 мс | 58.2 мс | 140.3× | 99.2% |

### 3. Абляция подслоёв (`o_proj` vs SwiGLU `down_proj`)

| Цель | Компиляция | Размер (FP32) | PPL ответа | Сдвиг PPL vs база |
|---|:---:|:---:|:---:|:---:|
| База (без адаптера) | — | — | 86.124 | 0.00% |
| Внимание (`o_proj`) | **413 мс** | **898 KB** | 77.361 | −10.17% |
| MLP SwiGLU (`down_proj`) | 6702 мс | 2882 KB | **75.634** | **−12.18%** |
| Dual (`o_proj` + `down_proj`) | 7017 мс | 3780 KB | 77.842 | −9.62% |

`down_proj` отдельно даёт наибольшее падение перплексии; комбинация целей **не** монотонно лучше. Ни один весовой вариант не даёт дословного recall (см. §1).

### 4. Single-Needle стирание контекста (gated e2e-гард)

[`tests/test_e2e_erasure.py`](tests/test_e2e_erasure.py) закапывает один секретный код на глубине ~30% в 2898 токенах русского filler, стирает контекст и проверяет, что query-aware residual-KV префикс восстанавливает его дословно:

| Метрика | Значение |
|---|---:|
| Токенов контекста (стёрто) | 2898 |
| Токенов запроса | 45 |
| Токенов resident-префикса | 256 |
| Полный KV-кэш | 33.96 MiB |
| KV resident-префикса | **3.00 MiB** |
| Сжатие KV | **91.2%** |
| **Recall Residual-KV** | **100%** (assert) |
| Recall Pure-LoRA (диагностика) | 0% |

Реальные выводы жадного декодирования (код `428719`, стёрт из промпта):
- **Полный контекст (ceiling):** `428719` ✓
- **Query-aware residual-KV:** `428719` ✓ (дословно, контекст стёрт)
- **Pure-LoRA:** `12345678` ✗ (претрейн prior)

Запуск: `ENGRAM_RUN_E2E=1 pytest tests/test_e2e_erasure.py -m e2e -s`. Query-agnostic окно по хвосту контекста **теряет эту иглу (0%)** — только окно=запрос её удерживает. Это регрессионный гард fact-уровня «никаких 0%».

---

## 🔍 Научный анализ и известные ограничения

### 1. Почему recall чистых весов LoRA = 0/5 (информационный предел)
- **Доминирование prior ($W_{\text{base}} \gg \Delta W$):** при rank $r=16$, $\alpha/r=1.0$ норма $\|\Delta W\|_F$ мала относительно $\|W_{\text{base}}\|_F$. Логиты смещаются достаточно для сдвига перплексии, но жадный $\argmax$ всё равно выбирает высоковероятные претрейн-токены.
- **Несовпадение распределений ключей:** $\Delta W$ фитится на активациях документа $X_{\text{ctx}}$; на инференсе вход — короткий запрос $X_{\text{query}}$ в другом подпространстве, поэтому линейная карта не «срабатывает» на токенах запроса.
- **Ёмкость:** одна rank-16 линейная карта поверх 2000 токенов multi-topic прозы не может без потерь хранить несколько произвольных multi-token идентификаторов, извлекаемых из другого подпространства. Сравните с ROME/MEMIT, решающими $W_{\text{new}}K=V$ для *изолированных структурированных* троек субъект-отношение-объект. **Дословные факты требуют residual-KV уровня, а не только весов.**

### 2. Почему residual-KV multi-needle recall = 2/5, а не 5/5 (чувствительность к бюджету)
- SnapKV удерживает top-B токенов контекста, на которые смотрит запрос. При B=256 на 2000 токенов бюджет покрывает короткие факты (`6.85`, `3850 seconds`) точно, но **усекает длинные идентификаторы** (`CIPHER-7749-DELTA`, IP, `142.6`), искажая цифры.
- Это компромисс **бюджет/покрытие fact-span**, а не фундаментальный провал: увеличение B под полный fact-span или per-fact сохранение span повышает recall. Single-needle recall = 100% при том же бюджете, потому что весь бюджет концентрируется на одном факте.
- Честная позиция: мы **не** заявляем 100% multi-needle recall при жёстком бюджете. Мы заявляем *работающий дословный fact-уровень* (100% single-needle, 40% exact + 60% near-miss на сложном multi-needle), там где только веса дают 0%.

### 3. Природа ускорения TTFT
- Ускорение 35×–140× — целиком **усечение промпта** (16 токенов запроса или 256 токенов префикса вместо 2000 токенов контекста), не более быстрая матричная арифметика.
- Компиляция 0.4–7.9 с. Compile-and-query окупается только при повторных запросах к контексту. При наличии памяти **KV prefix caching** сохраняет 100% recall с тем же ~60 мс TTFT.

### 4. Обработка фаз RoPE (исправлено)
- Декодеры с Rotary Position Embeddings зависят от непрерывной угловой фазы $\theta_t = t \cdot b^{-2i/d}$. Наивное сжатие 2000 токенов в короткий префикс ломает непрерывность фаз → зацикливание/gibberish.
- **Фикс:** хранить ключи pre-RoPE ($K_{\text{store}} = R(-p_{\text{orig}})K$), выбирать (никогда не сливать) оригинальные токены и re-anchor при загрузке ($K_{\text{load}} = R(p_{\text{new}})K_{\text{store}}$) на позиции $0..B$. Выводы теперь связные (§1). Ключи *выбираются*, а не усредняются, потому что слияние ключей под RoPE фазово-некогерентно; значения можно сжимать.

---

## 📚 Связанные исследования

- **ROME / MEMIT** — Meng et al. (2022, 2023): аналитическая правка фактов в MLP через замкнутую форму МНК для изолированных троек субъект-отношение-объект.
- **AlphaEdit** — Sun et al. (2024): проекция в null-space для сохранения фоновых знаний при последовательных правках.
- **SnapKV** — Li et al. (2024): query-aware селекция KV по observation-window (основа Ур. 3).
- **DuoAttention / PyramidKV / KIVI / KVQuant** — разделение retrieval/streaming голов, пирамидальные бюджеты, квантование KV.
- **Context Distillation / Prompt Tuning** — Snell et al. (2022), Lester et al. (2021): дистилляция в soft-токены через backprop (EngramRS — zero-grad).
- **Cartridges / Doc-to-LoRA** — обучаемые per-document KV/LoRA через градиентный спуск.
- **KV Prefix Caching** — Kwon et al. (vLLM, 2023): переиспользование KV без потерь; baseline для сравнения.

---

## 🛠️ Архитектура и реализованные аудиты

```
EngramRS/
├── .github/workflows/ci.yml    # CI: fmt, clippy -D warnings, cargo test, maturin --release, pytest -m "not e2e"
├── Cargo.toml                  # Rust 1.81+, faer, rayon, ahash, safetensors, half, pyo3
├── pyproject.toml              # Maturin, CLI entry point [project.scripts] engram
├── src/                        # Rust математическое ядро (core/critical path)
│   ├── deduplication/          # fast_hash.rs: zero-alloc AHash, изотропный Gaussian SimHash (Box-Muller)
│   ├── math/                   # anchors.rs (rSVD), solvers.rs (Dual Woodbury Cholesky), factorize.rs
│   ├── rope_gqa.rs             # GQA head-mapping + RoPE invert/re-anchor (фазовый фикс Ур. 3)
│   ├── serializer/             # peft_config.rs (alpha_pattern, rank_pattern), safetensors.rs (FP16/FP32)
│   ├── simd/                   # centering.rs (AVX2/AVX-512 sink-centering, BF16/FP16 unpack)
│   ├── streaming/              # accumulator.rs (Joint Frequent Directions [X | gamma Y])
│   └── lib.rs                  # PyO3 интерфейс модуля
├── python/engram_rs/           # Тонкий Python-слой
│   ├── catchers/               # attention.py (o_proj basis alignment), mlp.py, base.py
│   ├── compiler.py             # EngramCompiler (multi-target, layer_indices, safe unwrap)
│   ├── prefix.py               # EngramPrefixTuner: query-aware SnapKV residual-KV (Ур. 3)
│   ├── cli.py                  # CLI `engram` (compile, merge, export-gguf, compare-rag, benchmark-niah)
│   ├── gguf.py                 # Pure-Python GGUF v3 экспорт (канонический blk.N.attn_output.lora_a)
│   ├── merge.py                # Zero-grad merge с точным alpha/r масштабированием
│   ├── null_space.py           # Dimension-aware null-space ортогональная проекция
│   └── benchmarks/             # rag_comparison.py (честная аналитическая cost-модель)
└── tests/                      # 54 Rust + 87 Python (86 CI + 1 gated GPU e2e)
```

### Аудиты и фиксы v0.3.0
1. **Query-aware SnapKV (Ур. 3):** observation-window = запрос, а не хвост контекста. Фиксит 0% recall закопанных игл у query-agnostic префиксов.
2. **RoPE invert/re-anchor:** pre-RoPE хранение ключей + фазовый re-anchor при загрузке (`src/rope_gqa.rs`); устраняет gibberish виртуального префикса.
3. **`o_proj` basis realignment:** хук на вход `self_attn.o_proj` (истинная multi-head конкатенация), а не pre-attention layernorm.
4. **Dimension-aware null-space:** проекция соответствует $d_{\text{in}}$ каждой цели; молчаливые пропуски — ошибка конфигурации.
5. **PEFT uniform scaling (`alpha_pattern`):** $\alpha_i = r_i$ на слой, устраняет $4\times$ variance масштабирования.
6. **Joint Frequent Directions:** Frobenius-сбалансированный joint-sketch $[X \mid \gamma Y]$, $\gamma = \sqrt{\frac{\|X\|_F^2/d_{\text{in}}}{\|Y\|_F^2/d_{\text{out}} + 10^{-12}}}$.
7. **Изотропный Gaussian SimHash:** стандартные $\mathcal{N}(0,I)$ гиперплоскости через Box-Muller (вместо LCG).
8. **PEFT state hygiene:** `inject_engram` выгружает прошлые адаптеры перед повторной обёрткой.
9. **FP16 хранение:** `write_safetensors_typed` вдвое уменьшает размер адаптера.
10. **Честная cost-модель:** `compare-rag` — аналитическая модель top-k RAG vs resident-prefix с вычисляемым breakeven (без сфабрикованных замеров).

---

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/qaddasd/EngramRS.git
cd EngramRS
pip install maturin
maturin develop --release   # release обязателен: debug faer ~100x медленнее
```

### 1. Query-aware residual-KV префикс (Ур. 3 — дословные факты)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_rs import EngramPrefixTuner

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", attn_implementation="eager"  # SnapKV нужен eager attention
).eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

tuner = EngramPrefixTuner(model=model, tokenizer=tokenizer, prefix_len=256, preserve_sinks=1)

question = "Какой секретный код запуска?"
chat_question = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
)

# Запрос — observation-window: fact-токены удерживаются.
prefix = tuner.compile_prefix(
    context_text=document_text, prefix_len=256, selection="snapkv", query_text=chat_question
)

# Стираем документ; генерируем из короткого resident-префикса + запроса.
q_inputs = tokenizer(chat_question, return_tensors="pt")
gen_inputs = tuner.prepare_generate_inputs(q_inputs["input_ids"], prefix, attention_mask=q_inputs.get("attention_mask"))
out = model.generate(**gen_inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 2. Компиляция контекста в LoRA (Ур. 1 — стиль/домен)

```python
from engram_rs import EngramCompiler

compiler = EngramCompiler(model, tokenizer)
adapter_dir = compiler.compile_text(
    context_text=document_text, output_dir="./engram_adapter",
    targets=("o_proj", "down_proj"), layer_indices=range(8, 16), rank=16,
)
model = EngramCompiler.inject_engram(model, adapter_dir)
# ВНИМАНИЕ: только веса НЕ воспроизводят дословные факты (0%); для фактов — Ур. 3.
```

### 3. Standalone merge модели (zero-grad)

```bash
engram merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter ./engram_adapter --output ./merged_model
```

### 4. GGUF экспорт для llama.cpp и Ollama

```bash
engram export-gguf --adapter ./engram_adapter --output ./adapter.gguf
```

- **llama.cpp:** `llama-cli -m model.gguf --lora adapter.gguf -p "Question: ...?\nAnswer:"`
- **Ollama:** `Modelfile` с `FROM qwen2.5:0.5b` + `ADAPTER ./adapter.gguf`, затем `ollama create my-engram-model -f Modelfile && ollama run my-engram-model`

### 5. Аналитическое сравнение стоимости с RAG

```bash
engram compare-rag --context-tokens 10000 --rag-topk 2000 --resident-prefix 256 --compile-ms 1600
```

---

## 🧪 Запуск тестов

```bash
cargo test --verbose                       # 54 Rust-теста
cargo clippy --all-targets -- -D warnings  # 0 warnings
pytest -v tests/ -m "not e2e"              # 86 Python-тестов (CI gate)

# Gated GPU end-to-end гард стирания (скачивает Qwen2.5-0.5B-Instruct):
ENGRAM_RUN_E2E=1 pytest tests/test_e2e_erasure.py -m e2e -s   # +1 e2e (100% single-needle recall)
```

---

## 📄 Лицензия

[MIT License](LICENSE).