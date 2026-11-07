# Бенчмарки EngramRS

Все цифры ниже получены реальными прогонами, а не оценками. Дата замеров: 2026-11-07.

## Окружение

| Компонент | Значение |
|---|---|
| CPU/OS | Windows 10 (10.0.19045), x64 |
| GPU | NVIDIA GTX 1650, 4096 MiB, CUDA |
| Rust | 1.95.0 (release: `opt-level=3`, `lto="fat"`, `codegen-units=1`) |
| Python | 3.10.11, `.venv --system-site-packages` |
| Ключевые crate | faer 0.19.4, rayon 1.10, safetensors 0.4, PyO3 0.22 |
| Ключевые pip | maturin 1.16.0, transformers 4.57.6, peft 0.17.1 |

## 1. Синтетический perf-benchmark (Rust)

Обязательный тест в [tests/test_rust_math.rs](../tests/test_rust_math.rs): **28 слоёв**, `d_model=2048`, `rank=16`, `max_anchors=256`. Бюджет — **400 ms**, тест ассертит его напрямую.

```bash
cargo test --release -- --nocapture
```

Последний прогон: **43.800 ms** (запас ~9×).

### История оптимизации

| Итерация | Время | Что изменилось |
|---|---:|---|
| Плотный primal `d×d` Cholesky + SVD | 63 289 ms | наивная реализация — FAIL бюджета |
| Dual Woodbury + QR-редуцированный SVD | ~30–46 ms | математически эквивалентная замена, PASS |
| + дедупликация (наивный HashMap) | 1 290 ms | регрессия: поиск по большому ключу в каждой ячейке |
| + дедупликация (индексный HashMap) | 45.963 ms | `HashMap<Vec<u32>, usize>` + `Vec<DuplicateGroup>` |
| Финальный прогон | **43.800 ms** | после clippy-правок |

Вывод: узкое место было алгоритмическим (`d×d` vs `p×p`), а не константным. Бюджет не ослаблялся.

## 2. E2E benchmark стирания контекста

Файл: [tests/test_e2e_erasure.py:124](../tests/test_e2e_erasure.py). Запускается только при `ENGRAM_RUN_E2E=1`:

```bash
ENGRAM_RUN_E2E=1 python -m pytest -q -s tests/test_e2e_erasure.py
```

Модель по умолчанию: `Qwen/Qwen2.5-0.5B-Instruct` (можно переопределить через env), контекст — 256 токенов с вымышленным фактом (код `7391-KZR`); вопрос не содержит ответа.

### Результаты (CUDA / GTX 1650)

| Метрика | Значение |
|---|---:|
| Compile time | **238.829 ms** |
| Prompt tokens (до → после) | 256 → 22 |
| KV cache контекста | 3.000 MiB |
| Размер LoRA-весов | 2.631 MiB |
| TTFT base | 178.927 ms |
| TTFT engram | **93.019 ms** |
| Exact fact recall | **0.0%** |
| Base target perplexity | 176.366730 |
| Engram target perplexity | 131.308197 |
| Perplexity drop | **25.548%** |

### Что измеряется

- **KV cache memory** — фактические байты тензоров кэша контекста;
- **LoRA weights** — размер `adapter_model.safetensors` на диске;
- **TTFT** — синхронизированный CUDA-замер до первого токена, base vs PEFT-адаптер;
- **Recall** — точное вхождение секретного кода в сгенерированный ответ;
- **Perplexity** — по замаскированному целевому ответу, с адаптером и без (адаптер disabled/enabled через PEFT).

### Что ассертится, а что нет

Ассертится: compile < 1000 ms, сокращение числа токенов, наличие артефактов. **Recall намеренно не ассертится** — это эмпирический результат исследования, а не инвариант кода.

### Интерпретация

- Сжатие контекста работает: 256 токенов → адаптер 2.631 MiB, TTFT почти вдвое ниже, перплексия целевого ответа падает на 25.5% — адаптер несёт информацию о контексте.
- **Но exact recall = 0%**: модель не воспроизводит секретный код дословно. Утверждать успешное «стирание контекста с сохранением знания» нельзя — это открытая проблема (см. README, «Известные ограничения»).

## Воспроизведение всего пакета проверок

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release -- --nocapture          # 21 тест
source .venv/Scripts/activate
python -m maturin develop --release
ENGRAM_RUN_E2E=1 python -m pytest -q -s      # 14 тестов, ~13.4 s + загрузка модели
```

Последний сводный прогон: Rust **21 passed**, Python **14 passed in 13.41s**.