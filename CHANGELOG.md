# Changelog

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), проект следует [Semantic Versioning](https://semver.org/lang/ru/).

## [0.3.0] - 2026-11-07

Мажорный релиз устранения дефектов второго порядка, масштабирования потоковой ковариации и полной активации E2E бенчмарка без единого пропуска.

### Добавлено

- **Раздельный Frequent Directions скетч** (`src/streaming/accumulator.rs`): Устранено искажение сингулярных чисел и затухание кросс-ковариации $X^T Y$ при асимметрии норм $\|X\|_F \approx 100 \cdot \|Y\|_F$ в широких SwiGLU слоях ($d=4864$). SVD выполняется строго над $X$, проекция $Y$ масштабируется компенсирующим множителем $\sigma_{X, i} / \tilde{\sigma}_{X, i}$, а нормальные уравнения решаются через Cholesky повышенной точности в $f64$.
- **Компенсация ковариационного сдвига MLP** (`python/engram_rs/catchers/mlp.py`, `catchers/__init__.py`): Добавлен перехват `post_attention_layernorm` и метод реконструкции локального промежуточного входа $H_{\text{mlp}}^{\text{local}} = \text{SwiGLU}(\text{LN}(h_{\text{mid}}^{\text{full}} - \Delta Y_{\text{ctx}}))$, устраняющий сдвиг распределения ключей между обучением адаптера и инференсом.
- **Причинное авторегрессионное согласование весов удивления** (`python/engram_rs/compiler.py`): Исправлен сдвиг токенов на +1 при взвешивании потерь Cross-Entropy $\mathcal{L}_{\text{CE}}(t+1)$ с причинным экспоненциальным сглаживанием $0.7 \cdot \mathcal{L}_{t+1} + 0.3 \cdot \mathcal{L}_t$.
- **Точное масштабирование $\alpha/r$ и `rank_pattern` при слиянии LoRA** (`python/engram_rs/merge.py`): Предварительное масштабирование факторов на $\sqrt{\alpha_{k, l} / r_{k, l}}$ перед QR+SVD и последующая перенормировка на $\sqrt{\alpha_{\text{out}} / r_{\text{out}}}$, сохраняющая точную амплитуду $\Delta W$.
- **Активация сквозного E2E бенчмарка без пропусков** (`tests/test_e2e_erasure.py`): Снят `skipif`, подтверждена работа на реальной модели `Qwen/Qwen2.5-0.5B-Instruct` на GPU (`cuda`). Зафиксировано время компиляции **609 мс** (< 1000 мс), сжатие промпта на **94%** (345 -> 22 токена), ускорение TTFT на **55%** (193.8 мс -> 87.5 мс) и освобождение **4.043 МиБ** KV-кэша.
- **Точечная трансляция слоев LoRA**: Поддержка `layer_indices` в Rust ядре и `layers_to_transform` в `adapter_config.json`, исключающая порчу нецелевых ранних слоев внимания.
- **Ускорение Randomized SVD якорей** (`src/math/anchors.rs`): Переход на однопроходную проекцию $\Omega$ + thin QR без избыточных степенных итераций, сокративший время факторизации с ~50 мс до 0.25 мс на слой.

### Изменено

- Полный набор тестов: **103 passed** (33 Rust теста + 70 Python тестов, **0 skipped, 0 failed**).
- Версии пакетов обновлены до `0.3.0`.

---

## [0.2.0] - 2026-11-07

Глубокая модульная реструктуризация Rust ядра и внедрение высокопроизводительных SIMD/стриминг алгоритмов.

### Добавлено

- **Модульная архитектура Rust ядра**: Выделены специализированные подмодули `src/deduplication/`, `src/math/`, `src/serializer/`, `src/simd/`, `src/streaming/`.
- **SIMD AVX2/AVX-512 центрирование и распаковка** (`src/simd/centering.rs`): Вычитание sink-токена и распаковка BF16/FP16 с нулевым копированием прямо в Rust.
- **Fast AHash и косинусный SimHash** (`src/deduplication/fast_hash.rs`): Устранение аллокаций в куче при дедупликации строк матриц $X$ и $Y$.
- **Потоковый аккумулятор ковариации с Frequent Directions** (`src/streaming/accumulator.rs`): Обработка неограниченного контекста с фиксированным бюджетом памяти $O(d^2)$.
- **Модульная система перехватчиков** (`python/engram_rs/catchers/`): Изолированные ловцы активаций для Attention (`attention.py`) и MLP/FFN (`mlp.py`).

---

## [0.1.0] - 2026-11-07

Первый рабочий релиз: полный цикл «активации → LoRA-адаптер» без `backward()`.

### Добавлено

- **Rust-ядро** (`src/math_core.rs`): аналитическая компиляция слоя — точная дедупликация строк с сохранением достатных статистик, randomized-SVD anchor-компрессия, ridge solve с нуль-регуляризацией (primal Cholesky и dual Woodbury), rank-r факторизация через QR + thin SVD с fallback для вырожденного QR.
- **Параллельная компиляция всех слоёв** через rayon с детерминированным порядком и ошибками вида `layer {i}: ...`.
- **PyO3-мост** (`src/lib.rs`): `compile_all_layers_to_safetensors` с zero-copy заимствованием NumPy-буферов (`PyReadonlyArray2<f32>`), валидацией contiguous/форм/списков и возвратом времени компиляции в ms.
- **Сериализатор** (`src/serializer.rs`): PEFT-совместимые `adapter_model.safetensors` (F32 LE, ключи `...self_attn.o_proj.lora_A/B.weight`) и `adapter_config.json`.
- **GQA/RoPE-модуль** (`src/rope_gqa.rs`): head mapping/repetition, RoPE apply/invert, 6 unit-тестов.
- **Хуки захвата** (`python/engram_rs/hooks.py`): `EngramContextCatcher` — один forward-прогон, per-batch sink-центрирование, tensor/tuple-выходы, GQA-метаданные, lifetime-безопасные NumPy-массивы, снятие хуков и очистка CUDA-кэша.
- **Высокоуровневый API** (`python/engram_rs/compiler.py`): `EngramCompiler.compile_text` и `inject_engram` (PEFT hot-swap), восстановление train-режима модели.
- **Тесты**: 21 Rust и 14 Python.
- **E2E benchmark стирания контекста** (`tests/test_e2e_erasure.py`): реальный Qwen2.5-0.5B-Instruct, замеры KV-памяти, размера адаптера, TTFT, recall и perplexity.
- **Документация**: README, `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`.
