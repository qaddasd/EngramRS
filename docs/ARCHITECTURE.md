# Архитектура EngramRS

Документ описывает внутреннее устройство транспилятора EngramRS: модульную архитектуру, математический конвейер, аналитические методы синтеза адаптеров и виртуальных KV-префиксов, а также механизмы защиты от катастрофического забывания.

## Карта модулей

| Модуль | Ответственность |
|---|---|
| [src/lib.rs](../src/lib.rs) | PyO3-мост: приём мульти-таргетных NumPy-массивов (`o_proj`, `down_proj`), проекция нуль-пространства, бенчмаркинг задержки |
| [src/math_core.rs](../src/math_core.rs) | Вся математика ядра: дедупликация, SVD-anchors, асимметричный primal/dual ridge solve, Spectral Energy Allocation, randomized SVD |
| [src/rope_gqa.rs](../src/rope_gqa.rs) | GQA head-mapping/repetition, RoPE apply/invert, автономные unit-тесты |
| [src/serializer.rs](../src/serializer.rs) | Запись `adapter_model.safetensors` и PEFT `adapter_config.json` с поддержкой мульти-таргетов и динамического `rank_pattern` |
| [python/engram_rs/hooks.py](../python/engram_rs/hooks.py) | `EngramContextCatcher`: захват промежуточных активаций $m(x)$ для `down_proj` и внимания `o_proj` за один forward-прогон |
| [python/engram_rs/compiler.py](../python/engram_rs/compiler.py) | `EngramCompiler`: высокоуровневый компилятор (мульти-таргеты, стриминг, spectral energy rank, PEFT injection) |
| [python/engram_rs/prefix.py](../python/engram_rs/prefix.py) | `EngramPrefixTuner`: спектральное и Nyström сжатие контекстных $K_C, V_C$ в виртуальный префикс со 100% сохранением софтмакса |
| [python/engram_rs/streaming.py](../python/engram_rs/streaming.py) | `StreamingCovarianceAccumulator`: потоковое чанкование по 2048 токенов с фиксацией памяти $O(d^2)$ для контекстов 100k+ токенов |
| [python/engram_rs/null_space.py](../python/engram_rs/null_space.py) | Strict Null-Space Orthogonal Projection $\Delta W (I - X_{null}^+ X_{null})$ и калибровка фоновых якорей |
| [python/engram_rs/gpu_solver.py](../python/engram_rs/gpu_solver.py) | Direct In-VRAM GPU solver на `torch.linalg` без перемещения данных через CPU шину (~10-20 мс) |
| [python/engram_rs/merge.py](../python/engram_rs/merge.py) | Zero-Grad Model Merge ($W_{new} = W_{base} + \alpha A B^T$) с экспортом в автономную Hugging Face модель |
| [python/engram_rs/gguf.py](../python/engram_rs/gguf.py) | Чистый Python GGUF v3 бинарный писатель LoRA для Ollama (`--lora adapter.gguf`) и llama.cpp |
| [python/engram_rs/cli.py](../python/engram_rs/cli.py) | Полнофункциональный консольный интерфейс `engram` (`compile`, `merge`, `export-gguf`, `benchmark-niah`, `compare-rag`) |
| [python/engram_rs/benchmarks/](../python/engram_rs/benchmarks/) | Научные замеры: Needle In A Haystack (NIAH) и аналитическое сравнение стоимости/TTFT против RAG |

---

## Математический конвейер

### 1. Адаптация MLP/FFN-блоков (Closed-Form Knowledge Injection)
В декодерах архитектур LLaMA/Mistral/Qwen промежуточное представление FFN:
$$m(x) = \text{SiLU}(W_{gate} x) \odot (W_{up} x)$$
проектируется через $W_{down} \in \mathbb{R}^{d_{model} \times d_{intermediate}}$ обратно в остаточный поток.
EngramRS захватывает входные матрицы $M \in \mathbb{R}^{N \times d_{in}}$ и целевой остаточный выход $Y \in \mathbb{R}^{N \times d_{out}}$ ($d_{in} = d_{intermediate}, d_{out} = d_{model}$) и находит смещение весов:
$$\Delta W_{down} = (Y - W_{down} M)(M^T M + \lambda I)^{-1} M^T$$
разлагая его через SVD в $lora_A \in \mathbb{R}^{r \times d_{in}}$ и $lora_B \in \mathbb{R}^{d_{out} \times r}$.

### 2. Spectral Energy Allocation (Адаптивный ранг)
Ранг слоя выбирается динамически по порогу накопленной дисперсии сингулярных чисел:
$$\frac{\sum_{i=1}^{r_l} \sigma_i^2}{\sum_{j} \sigma_j^2} \ge \tau \quad (\tau = 0.95)$$
В PEFT адаптер записывается словарь `rank_pattern`, позволяющий модели использовать разные ранги на разных слоях и проекциях без конфликтов.

### 3. Потоковая кумулятивная ковариация (Streaming Covariance)
Для текстов длиной $N > 100\,000$ токенов текст разбивается на чанки по $B = 2048$ токенов:
$$C = \sum_{b=1}^{K} X_b^T X_b \in \mathbb{R}^{d \times d}, \quad P = \sum_{b=1}^{K} X_b^T Y_b \in \mathbb{R}^{d \times d}$$
Память ограничена строго $O(d^2)$, независимо от длины документа.

### 4. Строгая ортогональная проекция в нуль-пространство (Null-Space Projection)
Для защиты от катастрофического забывания:
$$\Delta W \leftarrow \Delta W \cdot (I - X_{null}^+ X_{null})$$
где $X_{null}^+$ — псевдообратная матрица Мура-Пенроуза фоновых представлений. Для любого фонового вектора $x \in \text{span}(X_{null})$ отклик адаптера тождественно равен $0.0000$ ($< 10^{-6}$).

### 5. Аналитический Virtual KV-Prefix Tuning
Вместо аппроксимации экспоненциальных спайков внимания линейным сдвигом, матрицы $K_C, V_C$ сжимаются спектрально вдоль временной оси в $P \approx 32..64$ виртуальных векторов префикса $(K_{prefix}, V_{prefix})$ через SVD или Nyström-проекцию. Префикс подается непосредственно в `past_key_values`, сохраняя 100% нелинейности софтмакса.
