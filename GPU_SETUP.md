# 🎮 Настройка GPU для RAG-системы

## Проверка видеокарты

### Windows:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Linux/Mac:
```bash
nvidia-smi
```

## Установка CUDA версии PyTorch

### Для NVIDIA GPU (Windows):

1. **Проверь версию CUDA:**
   ```bash
   nvidia-smi
   ```
   Посмотри на "CUDA Version" (например, 12.1)

2. **Установи PyTorch с CUDA:**

   **CUDA 11.8:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   **CUDA 12.1:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Проверь установку:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Должно вывести `True`

## Рекомендуемые модели для GPU

| Видеопамять | Модель | Скорость |
|-------------|--------|----------|
| **2-4 GB** | Qwen2.5-0.5B | ⚡⚡⚡⚡⚡ |
| **4-6 GB** | Qwen2.5-1.5B | ⚡⚡⚡⚡ |
| **6-8 GB** | Qwen2.5-3B | ⚡⚡⚡ |
| **8-12 GB** | Qwen2.5-7B | ⚡⚡ |
| **12+ GB** | Qwen2.5-14B | ⚡ |

## Настройка в коде

В файле `book_rag.py` можно явно указать модель:

```python
llm_model_name: str = 'Qwen/Qwen2.5-3B-Instruct'  # Для 6GB+
llm_model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'  # Для 4GB+
llm_model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'  # Для 2GB+
```

## Ускорение квантованием (для больших моделей)

Для экономии памяти GPU можно использовать 4-битное квантование:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

self.llm_model = AutoModelForCausalLM.from_pretrained(
    self.llm_model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Ожидаемая производительность

| Модель | CPU (токенов/сек) | GPU (токенов/сек) |
|--------|-------------------|-------------------|
| 0.5B | 20-30 | 100-150 |
| 1.5B | 5-10 | 50-80 |
| 3B | 2-5 | 30-50 |
| 7B | 1-2 | 15-25 |

## Решение проблем

### "CUDA out of memory"
- Уменьши модель (3B → 1.5B → 0.5B)
- Используй квантование (см. выше)
- Закрой другие приложения использующие GPU

### "CUDA not available"
- Переустанови PyTorch с CUDA версией
- Проверь драйверы NVIDIA
- Перезагрузи компьютер

### Медленная генерация на GPU
- Убедись что модель на GPU: `print(next(model.parameters()).device)`
- Используй `torch.float16` вместо `torch.float32`
- Включи `device_map="auto"`
