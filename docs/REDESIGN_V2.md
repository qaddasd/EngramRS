# EngramRS v2 — полный план переработки

> Статус: план (research → stable foundation). Дата: 2026-11-11.
> Основан на полном аудите Rust-ядра, Python-слоя, бенчмарков и обзоре литературы (2022–2026).
> Все утверждения о текущем коде верифицированы по исходникам с указанием file:line.

---

## 0. Резюме и честные пределы

**Диагноз.** Текущий EngramRS v0.3.0 имеет 0% exact fact recall не из-за «недостаточной регуляризации», а из-за трёх независимых фундаментальных дефектов, каждый из которых сам по себе обнуляет recall:

1. **Вакуумная целевая функция.** Солвер решает `min ||X·ΔWᵀ − Y||²`, где `Y` — *полный* выход проекции (`src/math/solvers.rs:16-17`, `cross = XᵀY`). Нигде в коде нет вычитания вклада базовых весов (`W·X`). При корректном базисе это вырождается в «выучить саму W на подпространстве контекста» (бесполезно и вредно: удваивает выход проекции на контекстно-подобных входах); при текущем некорректном базисе — в бессмысленный cross-map. **Компилировать нечего: целевой сигнал не определён.**
2. **Activation/basis mismatch.** X для `o_proj` берётся с выхода `input_layernorm` (`python/engram_rs/catchers/__init__.py:238-244`), тогда как рантайм-вход `o_proj` — конкатенация голов attention. Fallback на истинный вход (`:256`) никогда не срабатывает, т.к. layernorm-hook заполняет ключ первым. README.md:132 утверждает обратное. Обученная карта применяется к другому базису.
3. **Prior dominance без компенсации.** `lora_alpha = r` ⇒ scaling = 1.0 (`src/serializer/peft_config.rs:92,96`), плюс `depth_scale = 1/√L` дополнительно attenuates амплитуду дельты (`python/engram_rs/compiler.py:204-206` → `src/math/factorize.rs:84`, для L=24 множитель ≈0.20). Никакой калибровки амплитуды под факт нет.

Дополнительно: streaming делает независимые forward'ы чанков без cross-chunk attention (`python/engram_rs/streaming.py:199`), GGUF-имена несовместимы с llama.cpp (`python/engram_rs/gguf.py:345-356`), virtual-prefix ломает RoPE-фазы (`python/engram_rs/prefix.py:545`), а e2e-тест **не ассертит recall** (`tests/test_e2e_erasure.py:188-190`) — поэтому 0% не ловится CI.

**Что математически невозможно (и что делаем вместо):**

| Требование | Вердикт | Лучшая достижимая альтернатива |
|---|---|---|
| 95–100% verbatim recall **только весами** после полного стирания, для произвольного контекста | **Невозможно гарантировать.** Ёмкость весов конечна: ~2 бит/параметр (Allen-Zhu & Li, «Physics of LLMs 3.3», 2024); Fano-граница `P_e ≥ (H(Y\|Q) − B − 1)/log\|Y\|` для любой памяти объёмом B бит. Rank-r дельта не вмещает энтропию ответов большого корпуса; извлекаемость требует парафраз при обучении. | **Гибрид обязателен**: веса — для ассоциативных/реляционных фактов и стиля; residual-KV — для verbatim-критичного. 95–100% verbatim recall достижим гибридом при сайзинге бюджета под энтропию (см. §6). |
| «Ничего никогда не ухудшается» для всех входов | **Недоказуемо** для произвольных OOD-входов. | Точная гарантия на калибровочном подпространстве (AlphaEdit-проекция: `ΔW·x = 0` для `x ∈ span(K₀)`) + статистические non-regression гейты + мониторинг OOD-дрейфа. Формулировка гарантии — §2.5. |
| Multi-hop композиция через вставку фактов в веса | **Не работает** (MQuAKE-Remastered, ICLR 2025; RippleEdits, TACL 2024): locate-then-edit не propagate логические следствия. | Multi-hop идёт через retrieval/residual-KV путь (композиция в контексте — родная сила attention, Sanford–Hsu–Telgarsky 2023). Веса — только одиночные факты. |
| Статические правки `k_proj` под RoPE | **Некорректно поставлено**: ключ факта, вращённый на позиции запроса `p_q`, «переезжает» вместе с запросом. Адрес факта становится позиционно-зависимым. | Fact-режим таргетит `q/v/o + gate/up/down`; `k_proj` исключён из fact-режима (документируется). KV-путь хранит pre-RoPE ключи и переприменяет RoPE при загрузке (§2.4). |
| Latency/VRAM «всегда лучше KV-cache» | Условно: выигрыш TTFT существует **iff** резидентная память ≪ исходного контекста. | Честный учёт: residual-prefix B токенов vs контекст N токенов, B ≪ N ⇒ TTFT ≈ KV-cache короткого префикса; merge адаптера в базу ⇒ zero overhead decode. §6 гейты. |
| Zero-grad closed-form как единственный механизм | ROME/MEMIT доказывают ~99% efficacy **одиночных** фактов closed-form; произвольное long-context поведение так не дистиллируется. | Closed-form — default (быстро); опциональная gradient-refinement стадия (self-study + logit-KL, Cartridges) для последних % и стиля. Честно маркируется как не-zero-grad. |

**Стратегия.** Превратить EngramRS из «LS-фиттера активаций» в **трёхуровневый компилятор памяти** с маршрутизацией, доказуемой защитой базы на калибровочном подпространстве, и residual-KV как гарантом verbatim recall. Ядро — Rust; Python — тонкий слой захвата/интеграции.

---

## 1. Глубокий анализ текущего состояния

### 1.1 Что измерено на самом деле (ground truth)

Единственная реальная модель — Qwen2.5-0.5B-Instruct, GTX 1650 4 GiB, fp16, greedy. Сырой артефакт: `benchmarks/complex_benchmark_results.json`.

| Метрика | Значение | Источник |
|---|---|---|
| Exact recall после стирания (Engram LoRA) | **0/5 (0%)**, генерации токен-в-токен совпадают с zero-context prior | JSON:29,62,95,128,161 |
| Full context | **4/5 (80%)**, не 100%: обрезка `max_new_tokens=24` (`benchmarks/run_complex_comparison.py:137`); README заявляет 5/5 — расхождение | JSON:47 |
| Virtual prefix (32 tok) | 0/5, garbage («:[[:[[ORSспект…») — RoPE-коллапс | JSON:34,67,… |
| Target PPL (full→zero→engram) | 1.461→85.855→85.665 (Δ −0.22%) и т.д. — дельта статистически неотличима от нуля | JSON:27,60,… |
| Абляция | o_proj −0.21%, down_proj −3.03%, dual **+0.23% (ухудшение)**; README приводит другие числа — расхождение | JSON:203-227 |
| Compile time | o_proj 407 ms; down_proj 6701 ms; dual 7139 ms @ ~2k tok; superlinear scaling: 591→1487 ms, 1156→3900 ms, 2128→7676 ms | JSON:176-223 |
| TTFT «speedup 13.7–92.2×» | Целиком от truncation промпта (16 vs 2000 токенов), не от ускорения математики (README.md:86-88 честно признаёт) | JSON:11,25 |
| Semantic similarity, multi-hop, non-regression | **Не измерялись никогда** | — |

Методологические дефекты eval: один фиксированный needle во всех 15 ячейках; substring-scoring вместо exact match; вопрос лексически повторяет needle (подсказка); 1 прогон на ячейку, без сидов/CI; `rag_comparison.py` — аналитический калькулятор с хардкодом compile=45 ms (против измеренных 407–7676 ms) и RAG-strawman без top-k; chat-template не применяется нигде (instruct-модели получают raw prompt — само по себе занижает recall); committed JSON не перегенерирован после переработки раннера.

### 1.2 Корневые причины 0% recall (верифицировано)

**M1. Вакуумный fit-target (critical, математика).** `src/math/solvers.rs:16-17`: `gram = XᵀX`, `cross = XᵀY`, Y — полный выход проекции. Grep по `src/` подтверждает: вычитания `W·X` нет нигде. Следствия:
- При истинном базисе `Y = W·X` тождественно ⇒ LS восстанавливает саму W на span(X). Дельта ≈ копия W ⇒ удвоение выхода на контекстно-подобных входах, ноль полезного сигнала для запросов.
- «Contrastive delta» из документации (`ΔY = Attn(full) − Attn(local)`) **не реализован**: второго local-forward нет, Y подменяется на sink-centered attn_out (`catchers/__init__.py:402-403`).
- Вывод: нужен **контрфактический residual** (teacher-with-context минус student-without-context на query-side состояниях) либо MEMIT-постановка (key→value-residual). См. §2.2–2.3.

**M2. Basis mismatch o_proj (critical).** `catchers/__init__.py:238-244` кладёт выход `input_layernorm` в `_inputs["o_proj"]`; guard `:256` (`if layer_index not in self._inputs["o_proj"]`) мёртв. Рантайм-вход o_proj — attention-concat (смесь V после RoPE). Карта `normed_hidden → attn_out` не применима к `attn_concat → attn_out`.

**M3. Prior dominance (critical).** scaling=1.0 + `depth_scale=1/√L` (≈0.20 при L=24) ⇒ `||ΔW|| ≪ ||W||`; калибровки «достаточно ли дельты, чтобы перебить prior» нет. Даже идеальная дельта не меняет argmax.

**M4. Query-distribution mismatch (critical, концептуальный).** Фит минимизирует ошибку на **контекстных** токенах прозы. В момент вопроса скрытое состояние `X_query` не похоже ни на один `X_ctx`; линейная карта, обученная на прозе, не кодирует «когда спросят passcode — выдай 92481-OMEGA». ROME/MEMIT достигают ~99% именно потому, что фитят на **query-side ключах** (репрезентация субъекта в вопросительной формулировке), а не на контекстной прозе.

**M5. Streaming без causal continuity (critical).** `streaming.py:199`: `model(input_ids=chunk_ids, use_cache=False)` — позиции сбрасываются, cross-chunk attention отсутствует; overlap=128 смягчает лишь локально. Ковариация копится по семантически неверным активациям. Тест эквивалентности кормит синтетику, минуя этот путь (`tests/test_streaming.py:47`).

**M6. RoPE phase collapse virtual-prefix (critical).** `prefix.py:545`: `cache_position = arange(past_len, …)` по **сжатой** длине, а сжатые ключи несут RoPE **исходных** позиций. Фазы не совпадают ⇒ garbage. Спектральное слияние ключей под RoPE в принципе некорректно (см. §2.4: select K, merge V).

**M7. GGUF naming (critical).** `gguf.py:345-356` ⇒ `blk.0.attn_output.weight.lora_a`; канон llama.cpp — `blk.0.attn_output.lora_a` (без `.weight`). llama.cpp/Ollama молча не находят тензоры. Runtime-валидации нет (только byte-roundtrip).

**M8. Тесты не ловят дефекты (critical, процесс).** e2e ассертит только `compile_ms < 1000`, `after_tokens < before_tokens`, наличие файла (`test_e2e_erasure.py:188-190`) — зелёный при 0% recall. `test_prefix.py:230` «softmax preservation» — тавтология (вызывает ручной `torch.softmax`, не код проекта). CI гоняет e2e без env-gate (маркер обещает `ENGRAM_RUN_E2E=1`, но переменная не читается), качает Qwen каждый прогон. GPU CI нет; perf-gate `<400 ms` под `#[cfg(not(debug_assertions))]` в CI не исполняется.

### 1.3 Что в текущем коде здорово (сохраняем)

- Primal/dual ridge-солверы математически корректны (push-through эквивалентность, QR при rank-deficiency, SVD-fallback с верным транспонированием) — `src/math/solvers.rs`.
- LoRA-факторизация `A=√s·U, B=√s·V` и PEFT-конвенция — корректны (`src/math/factorize.rs`, тест `apply_lora`).
- Null-space алгебра: f64 двухпроходная MGS, leakage <1e-6 (`src/math/mod.rs:133-182`); numpy-версия с реальными calibration anchors (`null_space.py`).
- RoPE half-split forward/inverse и GQA repeat_kv — корректны (`src/rope_gqa.rs`), но **мёртвый код** (не экспортирован, не вызывается).
- FD-shrinkage как таковой валиден (Liberty 2013); dense-аккумулятор O(d²) f64 — верная идея.
- Analytical merge (QR+SVD truncate, amplitude-preserving) — корректен; тест logits vs PEFT atol=1e-5 (`tests/test_merge.py:81`).
- GPU↔Rust equivalence <1e-4 (`tests/test_gpu_solver.py`).
- GGUF v3 binary layout (magic, alignment, ne_dims) — корректен; ломается только naming.
- Safetensors/PEFT-экспорт — совместим, реально грузится через `PeftModel.from_pretrained`.

### 1.4 Реестр дефектов

Полный реестр (~45 пунктов с severity, file:line, фиксом и фазой) — §8.

---

## 2. Целевая архитектура

### 2.1 Обзор: три класса памяти + маршрутизатор

```
                         ┌────────────────────────────────────────────┐
                         │              CONTEXT (100k–1M+)            │
                         └──────────────────┬─────────────────────────┘
                                            │ capture (один forward, hooks,
                                            │ streaming FD-sketch, KV-запись)
                                            ▼
                         ┌────────────────────────────────────────────┐
                         │   ROUTER (Rust): per-span scoring          │
                         │   verbatim-ness · surprise · retrieval-attn│
                         │   · entity-density · null-space capacity ρ │
                         └───────┬──────────────┬──────────────┬──────┘
                 verbatim/критич │              │ реляционные  │ стиль/тема
                                 ▼              ▼ факты        ▼
                    ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
                    │ RESIDUAL KV    │ │ FACT COMPILER  │ │ SOFT COMPILER  │
                    │ (hard memory)  │ │ (MEMIT-style,  │ │ (contrastive   │
                    │ retrieval heads│ │  closed-form,  │ │  residual fit, │
                    │ exact K,V;     │ │  AlphaEdit     │ │  adaptive rank,│
                    │ streaming heads│ │  protection)   │ │  rsLoRA)       │
                    │ compressed     │ │ q/v/o+gate/up/ │ │                │
                    │ pre-RoPE store │ │ down; NO k_proj│ │                │
                    └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                            │                  │                  │
                            ▼                  ▼                  ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  MEMORY STORE (Rust): append-only log + manifest    │
                    │  units: KvBlock | FactEdit | StyleAdapter           │
                    │  compose · unlearn · TTL · quarantine · rollback    │
                    └──────────────────────┬──────────────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  SERVING: [system][residual-KV prefix][query]       │
                    │  + hot-swappable LoRA (vLLM/S-LoRA) или merged base │
                    │  re-RoPE at load · prefix caching · GGUF/Ollama     │
                    └─────────────────────────────────────────────────────┘
```

**Принцип разделения** (обоснован литературой):
- **Retrieval heads** (<5% голов, Wu et al. ICLR 2025) делают copy-paste из контекста — их KV обязан остаться точным. **DuoAttention** (ICLR 2025): полный KV только retrieval-головам, streaming-головам — sinks+recent; 2.55× память без потерь.
- **Веса** — для ассоциативного/реляционного знания (ROME/MEMIT ~99% efficacy одиночных фактов) и стиля; ёмкость конечна, multi-hop не propagate.
- **Маршрутизатор** решает per-span; default при неуверенности — KV (безопасная сторона).

### 2.2 Fact Compiler (замена вакуумного LS)

**Постановка (MEMIT/EMMET + AlphaEdit).** Факт `f = (s, r, o)` (субъект, отношение, объект). Для целевых слоёв `ℓ ∈ [ℓ_lo, ℓ_hi]` (средняя треть модели — эмпирика MEMIT):

1. **Ключ** `k*_ℓ` = вход целевой проекции на последнем токене субъекта в **query-side формулировке** («The passcode of the vault is …»), захваченный отдельным коротким forward. Это лечит M4: ключ живёт в том же распределении, что и запрос.
2. **Целевой residual** `δ*_ℓ` — добавка к выходу проекции, делающая объект `o` argmax-победителем на probe-запросе. Полностью closed-form через logit-lens:
   - Запись в residual stream: `Δz_ℓ` (выход down_proj/o_proj пишется прямо в stream).
   - Линеаризованный эффект на логиты: `Δlogit ≈ W_U · J_ln(z_final) · Δz_ℓ`, где `J_ln` — якобиан финального RMSNorm на probe-состоянии (диагональ + поправка на норму, считается аналитически).
   - Направление: `d_ℓ = W_U[o] − W_U[t*]`, `t*` — текущий top-token prior. `δ*_ℓ = β_ℓ · d_ℓ` (для down_proj; для o/v — через соответствующий write-path).
   - **Калибровка усиления β (убийство prior dominance, M3):** бисекция/секущая по `β` так, чтобы `logit_o − max_{t≠o} logit_t ≥ m` (margin m ≈ 2–5) на probe. 2–4 коротких forward на факт — дёшево, измеримо, вместо магической α.
3. **Батчевый closed-form update** (n фактов, слой ℓ):

   ```
   K_ℓ = [k*_1 … k*_n] ∈ R^{d_in×n},  R_ℓ = [δ*_1 … δ*_n] ∈ R^{d_out×n}
   ΔW_ℓ = R_ℓ K_ℓᵀ (K_ℓ K_ℓᵀ + λ_ℓ C₀_ℓ)^{-1}        (C₀-взвешенный ridge, à la ROME)
   ΔW_ℓ ← ΔW_ℓ · P_ℓ,   P_ℓ = I − U₀_ℓ U₀_ℓᵀ          (AlphaEdit-проекция, §2.5)
   ```

   Многослойное распределение residual (MEMIT): жадно по слоям сверху вниз — каждый слой забирает долю residual пропорционально его write-strength `||W_U W_down_ℓ||`, остаток передаётся ниже; опционально joint-optimization целевых z (маленькая задача, только векторы).
4. **Экспорт**: rsLoRA-scaling `α/√r` (rank-инвариантная амплитуда, Kalajdzievski 2023) вместо α/r; фактическая величина уже вшита калибровкой β, поэтому export-scaling = 1 и `depth_scale` **удаляется**.
5. **Negative examples / contrastive**: ключи «негативных» фактов (то, что менять нельзя) добавляются в защищаемый набор K₀; парафразы факта (обязательны для извлекаемости — Physics of LLMs) генерируются self-study и включаются в K как аугментация.

**Таргеты**: `q_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. **`k_proj` исключён из fact-режима** (RoPE-позиционная некорректность, §0). GQA: захват per-KV-head, `v_proj` d_out = n_kv·d_head, `o_proj` d_in = n_q·d_head — сериализатор обязан поддерживать mixed shapes (фикс B6).

**Честный предел**: efficacy одиночных фактов ~99% достижима (ROME/MEMIT existence proof); multi-hop — нет (§0); при насыщении null-space (§2.5) факт маршрутизируется в KV.

### 2.3 Soft Compiler (стиль/тема/soft knowledge)

Единственная математически осмысленная замена вакуумного LS — **контрфактический residual на query-side распределении**:

```
probe-набор P: self-study вопросы/продолжения по контексту (генерирует сама модель)
для каждого probe p:
    teacher: forward([ctx; p])  → Y_ctx (выходы проекций на позициях p)
    student: forward([p])       → Y_base, X̃ (входы проекций на позициях p)
    ΔY = Y_ctx − Y_base          (истинный контекст-индуцированный residual)
ΔW = argmin Σ_p w_p ||X̃_p ΔWᵀ − ΔY_p||²_F + λ_null||ΔW K₀ᵀ||²_F + λ||ΔW||²_F
```

- Это и есть «contrastive compilation» из документации, но **реализованная честно** (два forward, выровненные позиции).
- `w_p` — surprise-веса (√w на X и Y, чтобы эффективный вес был w, а не w² — фикс B24).
- Adaptive rank per layer/target по спектральной энергии ΔY (energy_threshold + residual-bound метрика, фикс B10).
- Adaptive ridge: `λ = ε·tr(C)/d_in`, ε ∈ [1e-3, 1e-2] (фикс B5); solve в f64 через QR скетча, не normal equations в f32.
- Опциональная **gradient-refinement** стадия (не-zero-grad, маркируется): LoRA r=32–128, несколько сотен шагов KL(teacher || student) на self-study данных (Cartridges recipe). Включается, когда closed-form residual-fit не проходит recall-гейт.

### 2.4 Residual KV (гарант verbatim recall)

**Конструкция:**
1. Во время capture записываем per-token важность: retrieval-head attention mass (извлечение голов — по DuoAttention-калибровке или **query-aware** observation-window голосованию à la SnapKV: окно = **запрос**, не хвост контекста) + маркеры fact-span от экстрактора + surprise.
2. **Селекция, не слияние, для ключей**: top-B токенов на слой/голову (пирамидальный бюджет PyramidKV: нижние слои — больше). Слияние ключей под RoPE некорректно (фазы разных позиций) — поэтому K только **выбираются** (оригинальные токены), V можно сливать/сжимать (V позиционно-независим).
3. **Хранение pre-RoPE**: `K_store = R(−p_orig)·K_ctx` (инверсия RoPE — подключаем мёртвый `src/rope_gqa.rs`), `V_store = V_ctx`. Квантование: K per-channel int4/int8 + outliers (KIVI/KVQuant), V per-token. Retrieval-головы — полная точность; streaming-головы — sinks+recent (DuoAttention).
4. **Загрузка**: `K_load = R(p_new)·K_store` — фазы всегда согласованы с новыми позициями (фикс M6). Префикс всегда занимает позиции `0..B` ⇒ это обычный короткий префикс ⇒ **нативный prefix-caching vLLM/SGLang работает из коробки**.
5. **Сшивка независимо захваченных блоков** (multi-doc): CacheBlend-подобный selective recompute ~10–20% токенов для cross-chunk attention вместо полного пересчёта.
6. Опционально (фаза 3): **cartridge** — обучаемый сжатый KV (self-study + logit-KL, Cartridges: 38.6× память, 26.4× throughput при качестве full ICL) для дальнейшего сжатия B.

**Бюджет**: B токенов residual-KV сайзится под энтропию ответов корпуса (Fano-оценка `H(Y|Q)` по probe-набору) и per-layer пирамиду. Гейт: B ≤ N/10 при recall ≥95% (N — длина исходного контекста).

**Критическое уточнение (валидировано empirically, GTX 1650 / Qwen2.5-0.5B):** observation-window ОБЯЗАН быть запросом (query-aware SnapKV). Фиксированный query-agnostic префикс (окно = последние 32 токена контекста) **теряет закопанные needle**: на 2k-токенном 5-needle haystack при бюджете 256 — **0/5 recall** (модель уверенно галлюцинирует значения). Query-aware селекция (окно = вопрос) даёт **100% verbatim recall** needle на глубине ~30% в 2898 токенах при сжатии до 256 (**91.2% KV, 11.3×**). Следствия: (a) «resident prefix» как фиксированный query-agnostic артефакт — это soft/style tier, он НЕ гарантирует verbatim recall; (b) для гарантий фактов префикс компилируется **per-query** (либо полные KV хранятся и сжимаются query-aware на инференсе — это и есть SnapKV); (c) окно селекции должно совпадать с запросом генерации (chat-template); (d) бюджет B должен покрывать ВСЕ токены fact-span — multi-token числа/коды особенно чувствительны к потере отдельных цифр (наблюдалась транспозиция цифр при недостаточном бюджете). Селекция K — per-layer top-B с сортировкой по исходной позиции (порядок needle сохраняется); слияние ключей под RoPE запрещено.

### 2.5 Защита базы: умный null-space (вместо «убивающего» проектора)

**Проблема текущего**: hard-проекция вычитает компоненты без рефита и без метрики потери сигнала; null молча отключается для down_proj при d_in≠d_null (`src/lib.rs:346-350`); в streaming проекция применяется дважды (`streaming.py:266-268,293-295`).

**Дизайн (AlphaEdit, ICLR 2025 Outstanding Paper + BetaEdit history-awareness):**

```
C₀_ℓ = Σ_{x ∈ calibration} x xᵀ      (FD-stream, f64; корпус + sink + история правок)
C₀_ℓ = U Σ Uᵀ  →  U₀ = top directions, покрывающие 99% энергии
P_ℓ = I − U₀ U₀ᵀ
ΔW_ℓ ← ΔW_ℓ P_ℓ
```

- **Точная гарантия**: `∀x ∈ span(U₀): ΔW·x = 0` — выходы базы на калибровочном распределении не меняются *тождественно*, не «приблизительно». Сигнал не умирает, потому что новые ключи K₁ имеют компоненты **вне** span(K₀): проекция срезает только ту часть апдейта, которая ломала бы старое (механизм AlphaEdit: +36.7% к SOTA при sequential editing).
- **Soft-версия**: штраф `Σ_i λ_i ||ΔW u_i||²`, `λ_i = λ_null·(σ_i/σ_max)^p` — сильные направления защищаем жёстко, слабые — мягко; калибровка p по non-regression гейту.
- **Монитор насыщения** (лечит «null-space съедает сигнал»): `ρ(k*) = ||P k*||²/||k*||²`. Если ρ < τ (≈0.1) — факт не вставить без касания защищаемого span ⇒ **маршрутизация в KV**, а не ослабление проекции. Это ранний индикатор «компилятор исчерпан» (null-space saturation, BetaEdit).
- **History-awareness**: ключи каждого скомпилированного юнита добавляются в K₀ ⇒ новые правки не интерферируют со старыми; пересчёт P инкрементальный (ключей мало — SVD маленькой матрицы).
- **Per-target размерности**: null считается в пространстве входов каждого таргета (down_proj: d=intermediate) — молчаливый пропуск запрещён, ошибка конфигурации вместо warning.
- **Честная формулировка гарантии**: точное сохранение на span(K₀); покрытие span оценивается по калибровочному корпусу; OOD-дрейф ограничен эмпирически гейтами (§6), не теоретически. BetaEdit документирует leakage при масштабе — мониторим и перекомпилируем.

### 2.6 Streaming и инкрементальная компиляция (100k–1M+)

- **Causal continuity**: инкрементальный forward с `DynamicCache` + сквозные `cache_position` (фикс M5); для независимо захваченных шардов — CacheBlend selective recompute.
- **Sketching**: Frequent Directions с **фиксированным γ** (выбирается один раз по pilot-чанку; текущий пересчёт γ при каждой компрессии ломает заявленные spectral bounds — `accumulator.rs:110-114`). Альтернатива: раздельные FD для X и Y + cross-covariance через joint-sketch с фиксированным γ. Гарантия FD: `||AᵀA − BᵀB||₂ ≤ ||A−A_k||_F²/(ℓ−k)`, mergeable — слияние шардов «бесплатно».
- **Точность**: накопление f64, solve f64 (фикс f64→f32 каста перед Cholesky, `accumulator.rs:384-407`); finite/overflow guards в compress (clamp γ, проверка joint-матрицы).
- **Память**: O(m·(d_in+d_out)) на sketch; q/k/v делят один X-sketch (post-LN hidden), gate/up — другой ⇒ 4 sketch'а на слой вместо 7. Пирамидальный m по слоям. Бюджет — явный параметр.
- **Скорость**: убрать `empty_cache()` per-hook (`catchers/base.py:130-132` — 32+ CUDA-синхронизации за compile); bulk-copy вместо поэлементного `Mat::from_fn`; rank-1 update вместо d×d аллокации на микро-чанк; rSVD с 1–2 power iterations (HMT/Musco–Musco) вместо наивного uniform-Ω без итераций (`anchors.rs:42-82`).
- **Инкрементальность**: новый контекст → новый юнит (O(new tokens)); merge аккумуляторов; сериализация состояния sketch'а; forgetting decay (Titans-style) для устаревших фактов.
- **Целевая цена**: sketch+solve ≤ 1 s @ 32k токенов CPU; capture-forward — нижняя граница (зависит от модели). Текущие 7.6 s @ 2k — устраняются фиксами выше.

### 2.7 Serving и интеграции

- **vLLM/SGLang**: скомпилированные дельты — hot-swappable LoRA per-request (нативный multi-LoRA vLLM; S-LoRA unified paging для тысяч адаптеров); residual-KV — обычный короткий префикс с нативным prefix-caching. Merge в базу (dLoRA credit-based) — только single-tenant steady state ⇒ zero decode overhead.
- **llama.cpp/Ollama**: GGUF LoRA с исправленным naming (`blk.{i}.attn_output.lora_a`), **runtime-валидация в CI** (реальная загрузка и генерация, не byte-roundtrip). Residual-KV — через session-state файл или короткий текстовый префикс (честно документируем ограничение).
- **HF transformers/PEFT**: `EngramMemory.apply(model)` → PeftModel + DynamicCache-префикс; `merge_and_unload()` для стационарного режима.
- **CLI**: `engram compile | serve | merge | benchmark | memory add/list/unlearn/validate | profile | doctor`. `serve` — OpenAI-compatible endpoint (vLLM backend, HF fallback). Chat-template применяется везде (фикс занижения recall).

### 2.8 Persistent memory store

```
store/
  manifest.json          (атомарный swap, версия схемы)
  log.jsonl              (append-only: unit_id, type, provenance, timestamp, TTL)
  units/
    <id>.facts.safetensors   (ключи, residuals, слои, gain, probe-метрики)
    <id>.kv.safetensors      (pre-RoPE K, V, quant-meta, позиции, head-map)
    <id>.style.safetensors   (LoRA + calibration stats)
```

- **Multi-document/multi-session**: каждый документ/сессия — юнит; композиция через subspace-aware merging (ортогонализация новых ключей против существующих; **не** наивный TIES/DARE — литература 2026 показывает pervasive sign conflicts).
- **Incremental updates**: append юнита без перекомпиляции; перекомпиляция только при насыщении ρ-монитора.
- **Unlearn**: точное вычитание дельты (линейность) или replay лога без юнита; KV-блок просто удаляется.
- **Catastrophic forgetting score**: sequential N юнитов → дрейф base-метрик + recall старых юнитов (§6 гейт).
- **Poisoning defense**: provenance каждого юнита; validation-probes при добавлении (меняет ли юнит ответы на калибровочные вопросы? противоречит ли существующим фактам?); quarantine + rollback по логу; per-unit gain caps.
- **Continuous learning**: режим append-only с TTL/decay и периодической консолидацией (replay → один merged adapter + компактный KV).

### 2.9 Диагностика

- `engram profile`: per-layer/per-target fit-residual, gain β, ρ-насыщение, спектральная энергия, время по стадиям.
- **Recall attribution**: абляция одной дельты/одного KV-блока → измерение падения recall на probe-наборе ⇒ «какой слой виноват».
- `engram explain <fact>`: какой юнит отвечает, весовой или KV-путь, уверенность, provenance.
- `engram doctor`: проверка basis-контракта (истинные входы проекций), dtype/device, версий transformers/PEFT, GQA-конфига.

---

## 3. Структура репозитория (Rust workspace + тонкий Python)

```
EngramRS/
  Cargo.toml                  # workspace
  crates/
    engram-core/              # ВСЯ математика, без pyo3, no_std-совместимое ядро где возможно
      src/
        lib.rs
        types.rs              # TargetSpec, LayerSpec, GqaConfig, RopeConfig, Budget
        sketch/
          fd.rs               # FrequentDirections (fixed-γ joint + per-matrix), mergeable
          rsvd.rs             # randomized SVD + power iterations, oversampling
          leverage.rs         # leverage-score / count-sketch sampling
        solve/
          ridge.rs            # f64 QR-solve, adaptive λ, primal/dual
          fact.rs             # MEMIT/EMMET batched equality-constrained update
          gain.rs             # β-калибровка (секущая/бисекция по probe-логитам)
        protect/
          nullspace.rs        # C₀ eigendecomp, P, soft/hard, ρ-монитор, history
        rope.rs               # apply/invert, phase-aware, partial rotary, interleaved
        gqa.rs                # repeat/expand, per-head views, zero-copy Cow
        factorize.rs          # SVD→LoRA, rsLoRA scaling, energy-rank + residual bound
        routing.rs            # span scoring, budget allocator (knapsack-greedy)
        capacity.rs           # Fano/entropy sizing, 2bits/param budget accounting
    engram-store/             # сериализация и persistent memory
      src/
        safetensors.rs        # mixed-shape, F32/F16/BF16 + range checks
        gguf.rs               # канонический naming, runtime-validated
        peft.rs               # rank/alpha patterns, все 7 таргетов
        manifest.rs           # атомарный manifest, схема, миграции
        log.rs                # append-only log, replay, rollback
        quant.rs              # KIVI/KVQuant-style K/V quantization, outliers
    engram-ffi/               # pyo3-биндинги (тонкие), GIL-release на копирующих путях
    engram-gpu/               # OPTIONAL: candle/cudarc kernels (sketch, solve, rope)
                              # feature-gated; CPU fallback всегда
  python/engram/
    __init__.py               # EngramCompiler, EngramMemory (публичный API)
    capture/                  # hooks: ИСТИННЫЕ входы проекций, GQA/RoPE-контракт
    extract/                  # fact extraction (self-extraction + heuristics), probes
    route.py                  # тонкая обёртка над core.routing
    compile.py                # оркестрация: fact/soft/kv pipelines
    memory.py                 # EngramMemory: add/apply/compose/unlearn/validate
    serve.py                  # OpenAI-compatible server (vLLM/HF)
    integrations/             # transformers, peft, vllm, sglang, llamacpp, ollama
    eval/                     # niah, ruler, nolima, mquake, longmemeval, nonreg, perf
    cli.py                    # engram compile|serve|merge|benchmark|memory|profile|doctor
  benchmarks/                 # reproducible runners + committed artifacts (regen!)
  tests/
    rust/                     # math vs references, property tests (FD bound, RoPE vs HF)
    python/                   # unit/integration; e2e С RECALL-АССЕРТАМИ
    ci/                       # gate-конфиги, пороги
  docs/                       # архитектура, контракты, honest limitations
  .github/workflows/          # cpu-matrix, gpu-nightly, benchmarks, release
```

Правила: вся критическая математика — в `engram-core` (тестируется без Python); Python не содержит линейной алгебры горячего пути (только захват тензоров и оркестрация); `engram-ffi` — единственный pyo3-слой; GPU — опциональная feature, CPU fallback обязателен.

---

## 4. Ключевые алгоритмы: псевдокод и Rust-сигнатуры

### 4.1 Fact compilation (batched, closed-form)

```rust
// engram-core/src/solve/fact.rs
pub struct FactSpec {
    pub keys: Mat<f64>,        // [n_facts, d_in]  query-side ключи субъекта
    pub residuals: Mat<f64>,   // [n_facts, d_out] калиброванные δ*
    pub paraphrase_keys: Option<Mat<f64>>, // аугментация для извлекаемости
}
pub struct Protection { pub u0: Mat<f64>, pub eigenvals: Vec<f64> } // span(K₀)

/// ΔW = R Kᵀ (K Kᵀ + λ C₀)⁻¹, затем ΔW ← ΔW·P.  f64, QR-based.
pub fn compile_facts(
    facts: &FactSpec,
    protect: &Protection,
    lambda_rel: f64,           // λ = lambda_rel · tr(C₀)/d
    mode: ProjectionMode,      // Hard | Soft { p: f64 }
) -> Result<DeltaW, CoreError>;

/// ρ(k*) = ||P k*||² / ||k*||² — маршрутизация в KV при ρ < τ.
pub fn insertion_capacity(keys: &Mat<f64>, protect: &Protection) -> Vec<f64>;
```

```python
# python/engram/compile.py (оркестрация, тонкая)
for fact in extractor.facts(context):            # self-extraction + heuristics
    probes = extractor.query_probes(fact)        # query-side формулировки + парафразы
    k = capture.keys(fact.subject, probes, layers)   # истинные входы проекций
    delta = gain.calibrate(fact, probes, layers)     # β: logit_o − logit_t* ≥ m, 2–4 forward
    rho = core.insertion_capacity(k, protection)
    if rho < TAU: router.to_kv(fact)             # null-space насыщен → KV
    else:         compiler.add_fact(fact, k, delta)
deltas = core.compile_facts(batch, protection, lambda_rel, mode)
```

### 4.2 Gain calibration (убийство prior dominance)

```rust
// engram-core/src/solve/gain.rs
/// Находит β: logit_o(probe; W + β·ΔW) − max_{t≠o} logit_t ≥ margin.
/// Секущая по β, 2–4 probe-forward (forward делает Python-слой, core — только математика).
pub struct GainCalibrator { pub margin: f64, pub max_beta: f64, pub iters: usize }
pub fn next_beta(cal: &GainCalibrator, history: &[(f64, f64)]) -> Option<f64>;
```

### 4.3 Streaming FD (fixed-γ, mergeable, f64)

```rust
// engram-core/src/sketch/fd.rs
pub struct FdSketch { ell: usize, d_in: usize, d_out: usize, gamma: f64, /* f64 buffers */ }
impl FdSketch {
    pub fn new(ell: usize, d_in: usize, d_out: usize, gamma: f64) -> Self; // γ фиксируется навсегда
    pub fn add_chunk(&mut self, x: &[f64], y: &[f64], n: usize) -> Result<(), CoreError>;
    pub fn compress(&mut self) -> Result<(), CoreError>;   // δ = σ²_{m+1}, shrink, finite-guards
    pub fn merge(&mut self, other: &FdSketch) -> Result<(), CoreError>; // шарды/сессии
    pub fn solve(&self, cfg: &SolveCfg) -> Result<DeltaW, CoreError>;   // f64 QR, adaptive λ
    pub fn spectral_error_bound(&self, k: usize) -> f64;   // ||A−A_k||_F²/(ℓ−k)
    pub fn save(&self, w: &mut impl Write) -> Result<(), CoreError>;    // инкрементальные сессии
    pub fn load(r: &mut impl Read) -> Result<Self, CoreError>;
}
```

### 4.4 Protection (AlphaEdit + history + saturation)

```rust
// engram-core/src/protect/nullspace.rs
pub struct NullGuard { u0: Mat<f64>, sigma: Vec<f64>, energy_covered: f64 }
impl NullGuard {
    pub fn from_calibration(sketch: &FdSketch, coverage: f64) -> Result<Self, CoreError>;
    pub fn protect(&self, dw: &mut DeltaW, mode: ProjectionMode);   // Hard: ΔW·P; Soft: штраф
    pub fn capacity(&self, keys: &Mat<f64>) -> Vec<f64>;            // ρ per key
    pub fn absorb_history(&mut self, compiled_keys: &Mat<f64>) -> Result<(), CoreError>;
    pub fn leakage(&self, probe_x: &Mat<f64>) -> f64;               // ||ΔW x||/||W x|| на probe
}
```

### 4.5 Residual KV (select K, merge V, re-RoPE)

```rust
// engram-store/src/quant.rs + engram-core/src/rope.rs
pub struct KvBlock {
    pub layer: usize, pub head_map: HeadMap,      // retrieval | streaming
    pub k_pre_rope: QuantTensor,                  // per-channel int4/8 + outliers
    pub v: QuantTensor,                           // per-token; слияние допустимо
    pub positions: Vec<u32>, pub token_ids: Vec<u32>,
}
pub fn invert_rope(k: &Mat<f32>, pos: &[u32], cfg: &RopeConfig) -> Mat<f32>;
pub fn apply_rope(k: &Mat<f32>, pos: &[u32], cfg: &RopeConfig) -> Mat<f32>;
pub fn select_tokens(scores: &HeadScores, budget: &PyramidBudget) -> Selection; // SnapKV-style
```

```python
# загрузка: фазы всегда согласованы
cache = DynamicCache()
for blk in store.kv_blocks():
    k = core.apply_rope(dequant(blk.k_pre_rope), new_positions, rope_cfg)
    cache.update(k, dequant(blk.v), blk.layer)
out = model.generate(query_ids, past_key_values=cache,
                     cache_position=arange(B, B+q))   # B = длина резидентного префикса
```

### 4.6 Router

```rust
// engram-core/src/routing.rs
pub struct SpanScore { pub verbatim: f64, pub surprise: f64, pub retrieval_attn: f64,
                       pub entity_density: f64, pub rho: f64 }
pub enum Route { Kv, Fact, Style }
pub fn route(span: &SpanScore, budget: &Budget) -> Route; // default Kv при неуверенности
pub fn allocate(spans: &[SpanScore], kv_budget: usize, rank_budget: usize)
    -> Vec<Route>; // greedy knapsack по score/cost
```

### 4.7 Ключевые уравнения (сводка)

| # | Уравнение | Назначение |
|---|---|---|
| E1 | `ΔW = R Kᵀ(K Kᵀ + λC₀)⁻¹`, `ΔW ← ΔW·P`, `P = I − U₀U₀ᵀ` | Fact insertion + точная защита базы на span(K₀) |
| E2 | `δ* = β·(W_U[o] − W_U[t*])`, β: `logit_o − max_{t≠o} logit_t ≥ m` | Closed-form value + калибровка против prior dominance |
| E3 | `ΔW = argmin Σ w_p‖X̃_pΔWᵀ − (Y_ctx−Y_base)_p‖² + λ_null‖ΔW K₀ᵀ‖² + λ‖ΔW‖²` | Soft/contrastive fit на query-side распределении |
| E4 | `λ = ε·tr(C)/d`; rsLoRA `α/√r` | Adaptive ridge + rank-инвариантная амплитуда |
| E5 | `‖AᵀA − BᵀB‖₂ ≤ ‖A−A_k‖_F²/(ℓ−k)` | FD-гарантия (fixed-γ!) |
| E6 | `K_store = R(−p_orig)K`, `K_load = R(p_new)K_store` | RoPE-корректный virtual prefix |
| E7 | `ρ(k*) = ‖Pk*‖²/‖k*‖²`, `ρ < τ ⇒ KV` | Монитор насыщения null-space |
| E8 | `bits ≈ 2·r·Σ_ℓ(d_in+d_out)`; `P_e ≥ (H(Y\|Q)−B−1)/log\|Y\|` | Сайзинг бюджета памяти (Fano / 2bits-per-param) |

---

## 5. План миграции с текущего EngramRS

**Принцип: strangler pattern.** v0.3 API сохраняется как deprecated-фасад над v2; каждый этап оставляет репозиторий зелёным.

**Этап 0 —止血 (критические фиксы, без новой функциональности):**
1. `catchers/__init__.py`: X для o_proj = `inputs[0]` o_proj-hook безусловно; удалить layernorm-как-источник-X и мёртвый guard. Basis-контракт тест (mock, где layernorm-out ≠ o_proj-in).
2. Удалить `depth_scale` из амплитуды; экспорт scaling=1 (величина — в калибровке).
3. `streaming.py`: инкрементальный forward с DynamicCache + cache_position.
4. `gguf.py`: naming без `.weight`; CI-тест реальной загрузки в llama.cpp.
5. `prefix.py`: хранить pre-RoPE K, re-RoPE при загрузке; запретить слияние K.
6. `test_e2e_erasure.py`: `assert recall > 0` (после фиксов 1–2 он должен проходить на простом needle); env-gate `ENGRAM_RUN_E2E`; CI `-m "not e2e"` + отдельный e2e-job.
7. Убрать per-hook `empty_cache()`; починить GPU-утечку в `_release_captures`; `torch.load(weights_only=True)`.
8. Перегенерировать benchmark JSON текущим раннером; привести README/STATUS в соответствие с артефактами (4/5 vs 5/5, абляция, B1-claim).

**Этап 1 — новое ядро:** workspace-разделение (core/store/ffi); перенос солверов в f64 QR; FD fixed-γ; подключение rope/gqa; fact-compiler + gain calibration + protection; router v0; residual-KV v0. Старый `compile_all_layers_to_safetensors` становится фасадом над soft-compiler.

**Этап 2 — память и serving:** memory store (log/manifest/compose/unlearn); CLI serve; vLLM/llama.cpp интеграции с runtime-валидацией; non-regression CI-гейты.

**Этап 3 — качество и масштаб:** self-study refinement, cartridge, GPU-path, quantization-aware compilation, 7B–70B матрица, streaming 100k–1M.

**Этап 4 — frontier:** MoE, continuous learning consolidation, poisoning hardening, RAG 2.0 API.

Совместимость: старые адаптеры грузятся (PEFT-формат не меняется); старые `compile()`-вызовы маппятся на soft-режим с предупреждением о смене семантики; формат store версионируется с миграциями.

---

## 6. Бенчмарки и критерии успеха (жёсткие гейты)

### 6.1 Recall-гейты (после полного стирания контекста)

| Тест | Порог (hybrid) | Порог (weights-only, честно) |
|---|---|---|
| NIAH single-needle, 1k–128k, ≥30 trials/ячейка, сиды, CI | **≥95%** exact match | ≥90% efficacy на одиночных фактах |
| RULER multi-key/multi-value/multi-query | **≥95%** | отчёт, без обещаний |
| NoLiMa (латентные ассоциации, без literal cues) | ≥90% от full-context baseline | отчёт |
| Verbatim probes (коды, числа, имена — regex-exact) | **≥99%** при B ≤ N/10 | не гарантируется |
| Multi-hop: MQuAKE-Remastered, BABILong | ≥80% от full-context baseline | не обещается (§0) |
| Agent memory: LongMemEval (update/abstention), LoCoMo | ≥90% extraction; update/abstention ≥85% | — |

### 6.2 Non-regression гейты (база с применённой памятью vs чистая база)

| Метрика | Порог |
|---|---|
| MMLU, GSM8K, HumanEval, HellaSwag | падение ≤0.5 pt absolute (или внутри 95% CI нуля) |
| HELMET, LongBench v2 (downstream long-context) | ≥99% от базы |
| Perplexity на held-out корпусе | ≤ +1% |
| Sequential forgetting: 1000 юнитов | base-дрейф ≤1 pt; recall старых юнитов ≥90% |
| Null-leakage на calibration span | <1e-6 (hard) — property-тест |

### 6.3 Performance-гейты

| Метрика | Порог |
|---|---|
| TTFT (повторные запросы) | ≤ vLLM prefix-caching baseline при matched recall |
| Decode latency | ≤1.10× baseline (merged); ≤1.15× (unmerged fused LoRA) |
| VRAM резидентной памяти | ≤1/10 полного KV @128k при recall-гейте |
| Compile (sketch+solve) | ≤1 s @32k CPU; end-to-end ≤1.5× prefill |
| Incremental add | O(new tokens), без полной перекомпиляции |
| Throughput под concurrency | ≥ baseline при thousand-adapter paging (S-LoRA) |

### 6.4 Матрица моделей и протокол

Qwen2.5-0.5B/1.5B/3B/7B/14B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-v0.3; 70B spot-check (Llama-3.1-70B, арендованный A100/H100); MoE — Qwen3-30B-A3B/Mixtral (фаза 4). Greedy **и** sampling (t=0.7, pass@k для code); ≥30 trials/ячейка, bootstrap CI; chat-template везде; baselines: full-context, zero-context, vLLM prefix-caching, retrieval-RAG с top-k (честный, не strawman). Все артефакты коммитятся и перегенерируются CI; README-таблицы генерируются из JSON (никаких ручных чисел).

---

## 7. Поэтапный roadmap

| Фаза | Срок | Содержание | Exit-критерий |
|---|---|---|---|
| **0. Стабилизация** | нед. 1–2 | 8 критических фиксов (§5), recall-ассерты, CI-gates, регенерация артефактов | e2e recall >0 на простом needle; CI зелёный без сети; README=JSON |
| **1. MVP hybrid** | нед. 3–6 | workspace; f64 QR; FD fixed-γ; fact-compiler + gain + protection; router v0; residual-KV v0; eval-сьют | **≥95% single/multi-needle verbatim (hybrid)** на 0.5B–3B; weights-only ≥90% single-fact; non-regression ≤0.5 pt |
| **2. Memory + serving** | нед. 7–12 | store/log/compose/unlearn; CLI serve; vLLM/llama.cpp runtime-validated; streaming 100k; 7B–14B; GPU CI nightly | Все гейты §6 на 7B; TTFT ≤ prefix-caching baseline; 1000-юнит forgetting ≤1 pt |
| **3. Качество + масштаб** | нед. 13–18 | self-study refinement; cartridge; GPU-path (candle/cudarc); quant-aware; FP8; 70B spot; 1M streaming | NoLiMa ≥90% baseline; B ≤ N/10 @128k; compile ≤1 s @32k |
| **4. Frontier** | ongoing | MoE; continuous learning consolidation; poisoning hardening; RAG 2.0 API | LongMemEval update/abstention ≥85%; MoE recall-гейт |

---

## 8. Реестр дефектов и улучшений (верифицировано, с обоснованием)

Severity: C=critical, M=major, m=minor. Фаза — когда исправляется. Все file:line проверены по исходникам.

### 8.1 Критические

| ID | Место | Дефект | Фикс / обоснование | Фаза |
|---|---|---|---|---|
| C1 | `src/math/solvers.rs:16-17` | Fit-target = полный выход проекции, не residual ⇒ вакуумная компиляция (M1) | Fact/soft-постановки §2.2–2.3: контрфактический residual или key→value | 1 |
| C2 | `catchers/__init__.py:238-244,256` | X o_proj = layernorm-out ≠ рантайм-вход; guard мёртв; README противоречит | Истинный `inputs[0]` o_proj; basis-контракт тест | 0 |
| C3 | `peft_config.rs:92,96` + `compiler.py:204-206` + `factorize.rs:84` | scaling=1 + depth_scale=1/√L ⇒ prior dominance | Удалить depth_scale; gain-калибровка β под margin; rsLoRA | 0→1 |
| C4 | `streaming.py:199` | Независимые chunk-forward, нет cross-chunk attention | DynamicCache + сквозные cache_position | 0 |
| C5 | `gguf.py:345-356` | `blk.i.attn_output.weight.lora_a` ≠ канон llama.cpp | Naming без `.weight`; runtime-валидация в CI | 0 |
| C6 | `prefix.py:545` | RoPE phase collapse: сжатые K несут старые фазы, позиции новые | pre-RoPE store + re-RoPE at load; select K, merge V | 0→1 |
| C7 | `test_e2e_erasure.py:188-190` | Recall не ассертится ⇒ 0% не ловится | `assert recall ≥ threshold`; env-gate; CI-разделение | 0 |

### 8.2 Major

| ID | Место | Дефект | Фикс | Фаза |
|---|---|---|---|---|
| M1 | `accumulator.rs:110-114` | γ пересчитывается при каждой компрессии ⇒ FD-bound не обоснован | Fixed-γ (pilot) или per-matrix FD | 1 |
| M2 | `accumulator.rs:110-158` | Нет finite/overflow-guard в compress; γ→∞ при Y≈0 | Clamp γ + finite-check joint | 1 |
| M3 | `accumulator.rs:384-407` | f64→f32 каст перед Cholesky аннулирует точность | f64 solve | 1 |
| M4 | `solvers.rs:33-39`, `lib.rs:217` | Неадаптивный λ=1e-4 при ‖x‖²~1e3–1e6 | λ=ε·tr(C)/d | 1 |
| M5 | `serializer/mod.rs:51-55` | Uniform d_out блокирует mixed-target (q/k/v+o, gate/up) | Per-target shapes | 1 |
| M6 | `peft_config.rs:28-34` | Нет маппинга q/k/v/gate/up ⇒ PEFT молча игнорирует веса | Полная таблица sub_path | 1 |
| M7 | `lib.rs:346-350` | Null молча пропускается при d_in≠d_null (down_proj) | Per-target null; ошибка вместо пропуска | 1 |
| M8 | `anchors.rs:42-82` | rSVD без power iterations, uniform Ω, oversampling может обнулиться | Gaussian Ω + 1–2 power iter + p≥5 | 1 |
| M9 | `anchors.rs:12-17` | Усечение до k=256 без метрики потерянной энергии | Residual bound в отчёт | 1 |
| M10 | `catchers/base.py:130-132` | `empty_cache()` per-hook ⇒ 32+ CUDA-синхронизации | Один вызов в конце compile | 0 |
| M11 | `catchers/__init__.py:434-442` | GPU-утечка: sub-catchers не чистятся в `_release_captures` | Обнулять sink/raw tensors | 0 |
| M12 | `streaming.py:266-268,293-295` | Двойная null-проекция в strict-режиме | Проецировать один раз | 1 |
| M13 | `merge.py:310` | Линейное слияние без interference-resolution | Subspace-aware compose (ортогонализация ключей) | 2 |
| M14 | `gpu_solver.py:319-323,396` | Hardcoded Llama naming; `strict=False` молчит | Арх-агностик маппинг; strict-load | 1 |
| M15 | `lib.rs:484-491` | rope/gqa — мёртвый код, инверсия не применяется | Подключить в KV-путь и capture | 1 |
| M16 | `math/mod.rs:133` vs `compiler.py:227` | Rust hard-projection test-only; numpy-дубль в проде | Единая реализация в core | 1 |
| M17 | `streaming.py:202,342,389` | Streaming только o_proj | Все таргеты через shared sketches | 1 |
| M18 | `prefix.py` Cache API | `from_legacy_cache`/`len(cache)` — version-fragile | Версионный адаптер + тесты на pin-версиях | 1 |
| M19 | `compiler.py:113-114`, `prefix.py:136` | batch>1 не поддержан (logits[0]) | Явная поддержка batch или assert batch=1 | 1 |
| M20 | CI | Нет GPU CI; perf-gate не исполняется (debug); e2e без gate | GPU nightly; release-perf job; env-gate | 0→2 |
| M21 | `benchmarks/rag_comparison.py:28,58,160` | Калькулятор с хардкодом compile=45ms, RAG-strawman, зашитый вывод | Реальный RAG baseline (top-k) + vLLM prefix-caching | 1 |
| M22 | `niah.py:60-62,155,211,218` | Фиксированный needle, paragraph-depth, substring, 1 прогон | Рандомизация, token-depth, exact match, ≥30 trials, CI | 1 |
| M23 | `run_complex_comparison.py:137` | max_new_tokens=24 обрезает ответы (4/5 артефакт) | ≥64 токена; exact-match по полному ответу | 0 |
| M24 | README/STATUS vs JSON | Расхождения чисел (5/5 vs 4/5; абляция; B1-claim) | Генерация таблиц из артефактов | 0 |
| M25 | `catchers/__init__.py:402-403` | «Contrastive delta» не реализован (нет local-forward) | Честный двух-forward contrastive (§2.3) | 1 |
| M26 | `cli.py:297` | CLI default targets=o_proj ≠ compiler default | Синхронизировать | 0 |
| M27 | Отсутствует chat-template | Instruct-модели получают raw prompt ⇒ занижение recall | `apply_chat_template` везде | 0 |

### 8.3 Minor

| ID | Место | Дефект | Фикс | Фаза |
|---|---|---|---|---|
| m1 | `safetensors.rs:16-23`, `mod.rs:75-85` | F16 overflow→inf без range-check; BF16-output нет | Range-check + BF16 | 1 |
| m2 | `deduplication/mod.rs:69-84` | 1-bit LSH recall ~40% | Multi-probe LSH | 2 |
| m3 | `fast_hash.rs:5-36` | LCG+Box-Muller корреляции (Neave) | PCG/xoshiro | 1 |
| m4 | `simd/centering.rs:5-21` | Скаляр под видом SIMD; BF16-конверсия per-row | Intrinsics/hoist sink; переименовать честно | 1 |
| m5 | `math/mod.rs:30-31` | Поэлементная копия активаций внутри Rust | Bulk-copy | 1 |
| m6 | `accumulator.rs:336-353` | d×d аллокация на микро-чанк | Rank-1 update | 1 |
| m7 | `accumulator.rs:44,318` | Unchecked mul (overflow в release) | checked_mul | 1 |
| m8 | `compiler.py:149-163` | Surprise w применяется к X и Y ⇒ w² | √w на оба | 1 |
| m9 | `prefix.py:515,473` | `torch.load` без weights_only; save GPU-тензоров | weights_only=True; .cpu() | 0 |
| m10 | `merge.py:354,367,424` | base_config=configs[0]; только lora_A-ключи; np не импортирован | Явная проверка конфигов; все форматы ключей; import | 1 |
| m11 | `compiler.py:3,272` | Sequence не импортирован (introspection) | import | 0 |
| m12 | `lib.rs:158` | Streaming solve хардкодит anchors=256, λ_null=0 | Пробросить config | 1 |
| m13 | `streaming.py:260` | Python-стриминг на numpy, Rust-класс не используется | Единый core-путь | 1 |
| m14 | `test_rust_math.rs:618-621` | Wall-time assert flaky | Относительный порог / отдельный perf-job | 0 |
| m15 | `mlp_solver.rs:19-70` | Дубль generic-кода под видом SwiGLU-солвера | Удалить дубль; честный docstring | 1 |
| m16 | `types.rs:120-131`, `fast_hash.rs:122-141`, `catchers/__init__.py:82` | Мёртвый код | Удалить | 1 |
| m17 | `test_prefix.py:230` | Тавтологичный «softmax preservation» | Реальный тест compressed-vs-full attention | 1 |
| m18 | `test_streaming.py:47` | Эквивалентность на синтетике минует баг M5/C4 | Тест через accumulate_from_model | 0 |
| m19 | `test_killer_features.py` | Mock-only, contrastive не покрыт | Реальные tiny-модели с gate/up | 1 |
| m20 | `test_merge.py:160` | target_rank=total_r ⇒ truncation не тестируется | Тест с truncation | 1 |

### 8.4 Новые возможности (traceability к требованиям)

| Требование | Элемент дизайна | Фаза |
|---|---|---|
| Гибрид fact/style/KV + авторешение | Router §2.1, §4.6 | 1 |
| Adaptive λ, spectral rank, лучший sketching | E4, FD fixed-γ, rSVD+power | 1 |
| Contrastive + negative examples | §2.3 (два forward), K₀-негативы | 1 |
| Все 7 проекций, RoPE/GQA/SwiGLU | §2.2 (k_proj исключён обоснованно), rope/gqa подключены | 1 |
| Умный null-space + calibration | §2.5 AlphaEdit + ρ-монитор + history | 1 |
| Streaming 100k–1M, fixed budget | §2.6 | 2–3 |
| GPU-path Rust + CPU fallback | engram-gpu (candle/cudarc), feature-gated | 3 |
| Persistent/multi-doc/incremental/merge | §2.8 store | 2 |
| Non-regression + recall + perf гейты | §6 | 0→3 |
| FP16/BF16/FP8, quant-aware | quant.rs + compile против dequantized W_q | 3 |
| GGUF/safetensors/PEFT/vLLM экспорт | engram-store + runtime CI | 0→2 |
| Surprise+attention+gradient-proxy важность | Router scores | 1 |
| Soft erasure + residual tokens | KV-путь + TTL | 2 |
| Continuous learning | append-only + consolidation | 4 |
| MoE | route-aware capture, shared-expert edits | 4 |
| Poisoning defense | provenance + validation probes + quarantine | 2→4 |
| Профилировщик/диагностика | §2.9 | 2 |

---

## 9. Ключевые источники

- ROME (NeurIPS 2022, arXiv:2202.05262); MEMIT (ICLR 2023, arXiv:2210.07229); EMMET (EMNLP 2024, arXiv:2403.14236); PMET (AAAI 2024).
- **AlphaEdit** (ICLR 2025 Outstanding, arXiv:2410.02355) — null-space без убийства сигнала; BetaEdit (IJCAI 2026) — leakage и history-awareness; WISE (NeurIPS 2024) — side-memory+router.
- **Cartridges** (ICLR 2026, arXiv:2506.06266) — self-study + logit-KL, 38.6× память; Doc-to-LoRA (Sakana 2026) — existence proof verbatim recall до 40k; SHINE (ICML 2026) — hypernetwork context→LoRA (и его recall gap vs ICL — аргумент за гибрид).
- **Retrieval Heads** (ICLR 2025, arXiv:2404.15574); **DuoAttention** (ICLR 2025, arXiv:2410.10819); SnapKV (NeurIPS 2024); PyramidKV (arXiv:2406.02069); KIVI (ICML 2024); KVQuant (NeurIPS 2024); CacheBlend (EuroSys 2025 Best Paper).
- Frequent Directions (Liberty KDD 2013; Ghashami et al. 2016); HMT randomized SVD (2011); Musco–Musco power iteration (2015).
- **LoRA Learns Less and Forgets Less** (TMLR 2024, arXiv:2405.09673) — all-module targeting; LoRA Without Regret (Thinking Machines 2025); rsLoRA (arXiv:2312.03732); S-LoRA (MLSys 2024); dLoRA (OSDI 2024); TIES (NeurIPS 2023) / DARE (ICML 2024) — и критика sign-conflicts (2026).
- **Physics of LLMs 3.3** (ICLR 2025, arXiv:2404.05405) — 2 bits/param, парафразы обязательны; rate–distortion граница recall (arXiv:2607.08032); Sanford–Hsu–Telgarsky (NeurIPS 2023) — attention vs FFN для retrieval.
- **MQuAKE-Remastered** (ICLR 2025) / RippleEdits (TACL 2024) — multi-hop не propagate через правки; Gupta et al. (ACL 2024) — gradual→catastrophic forgetting при sequential editing.
- Eval: RULER (COLM 2024), NoLiMa (ICML 2025), HELMET (ICLR 2025), LongBench v2 (ACL 2025), ∞Bench, BABILong, LongMemEval (ICLR 2025), LoCoMo (ACL 2024).
- Titans (NeurIPS 2025) / Atlas (2025) — surprise/forgetting decay; SEAL (NeurIPS 2025); MemoryLLM (ICML 2024); Larimar (ICML 2024) — closed-form write во внешнюю память (ближайший родственник).

---

## 10. Итоговая формулировка гарантий (что подписываемся)

1. **Verbatim recall ≥95–100%** после полного стирания — **гибридом** при B ≤ N/10, на матрице моделей §6.4, с CI. Не «только весами для произвольного контекста» — это информационно невозможно.
2. **База не деградирует**: тождественно на span(K₀) (доказуемо), статистически на MMLU/GSM8K/HumanEval/HELMET (≤0.5 pt), с мониторингом OOD и forgetting при sequential-режиме.
3. **Latency/VRAM**: TTFT ≤ prefix-caching baseline при matched recall; decode ≤1.10× (merged); резидентная память ≤1/10 KV @128k. Выигрыш TTFT — от короткого резидентного префикса, честно.
4. **Rust-ядро**: вся математика в engram-core, f64, property-тесты против референсов, zero-copy на границах где возможно, GPU опционален.
5. **Всё остальное** (multi-hop через веса, k_proj-правки под RoPE, «никогда не ухудшается для всех входов», zero-grad как единственный механизм) — **не обещаем**, с обоснованием и лучшей альтернативой (§0).