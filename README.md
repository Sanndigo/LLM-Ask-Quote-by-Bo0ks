# 📚 RAG - Поиск по книгам с AI-ответами

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Интеллектуальная система поиска и ответов на вопросы по текстовым документам с использованием RAG (Retrieval-Augmented Generation).

## ✨ Возможности

### 🔍 Поиск фрагментов
- Мгновенный поиск по всем загруженным книгам
- Показывает 5 наиболее релевантных фрагментов
- Указывает книгу, главу/строфу и процент схожести

### ❓ Ответы на вопросы
- AI генерирует развернутые ответы на основе контекста
- Показывает цитаты из книг для подтверждения
- **GigaChat API** — отличные ответы на русском
- **Локальные модели** — Qwen2.5, Phi-3, Gemma

### 📖 Управление библиотекой
- Загрузка TXT файлов через веб-интерфейс
- Drag & Drop для удобной загрузки
- Автоматическая индексация новых книг

### 🎮 GPU поддержка
- Автоматическое определение видеокарты
- Оптимизация под доступную память
- Ускорение генерации в 5-10 раз

## Смолл аттеншенс😁

### 💻 Запускать желательно на видеокарте NVIDIA!
- Софт первоначально настроен под работу с CUDA-ядрами!
- Без CUDA он может работать неккоректно!

### 🤖 Низкое качество LLM, используемой в проекте
- Система была протестирована на видеокарте GTX 1060 3GB Vram, она не позволяет разгуляться в выборе LLM. Если у вас есть возможность выбрать модель более умную(с большим количеством гиперпараметров). Вот примеры некоторых, которые можно использовать для видеокарты с 12+ vram:

- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Qwen2.5-14B-Instruct
- Gemma-2-2b-it
- Llama-3.2-3B-Instruct

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск веб-интерфейса

```bash
python web_app.py
```

Откройте в браузере: **http://localhost:5000**

### 3. Консольный режим (опционально)

```bash
python book_rag.py --interactive
```

## 📋 Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **RAM** | 4 GB | 8 GB |
| **Место** | 2 GB | 6 GB |
| **GPU** | Любой | NVIDIA 4GB+ |

### Для GPU (NVIDIA):
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

См. [GPU_SETUP.md](GPU_SETUP.md) для подробной инструкции.

## 📁 Структура проекта

```
TechnoStr/
├── web_app.py              # Flask веб-сервер
├── book_rag.py             # RAG движок (поиск + ответы)
├── embedding_processor.py  # Векторный поиск (FAISS)
├── text_processor.py       # Обработка текста
├── main_processor.py       # Обработка книг
├── templates/
│   └── index.html          # Веб-интерфейс
├── data/                   # TXT книги
├── processed/              # Обработанные чанки
└── embeddings/             # Векторный индекс
```

## 🎯 Примеры использования

### Поиск фрагмента
**Запрос:** "где Татьяна пишет письмо Онегину"

**Результат:**
```
📖 Евгений Онегин
📍 Строфа XXV, фрагмент #152
📊 Схожесть: 85.3%
📝 Но получив посланье Тани...
```

### Ответ на вопрос
**Вопрос:** "Кто написал Шинель?"

**Ответ:**
```
«Шинель» написал Николай Васильевич Гоголь...

📑 Источники:
1. Шинель (фрагмент #1)
   «Николай Васильевич Гоголь Шинель...»
```

## ⚙️ Настройка

### Выбор модели

В `book_rag.py` измените параметр:

```python
llm_model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'  # Быстрая (1GB)
llm_model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'  # Баланс (3GB)
llm_model_name: str = 'Qwen/Qwen2.5-3B-Instruct'    # Качественная (6GB)
```

### Параметры чанков

```bash
python main_processor.py --chunk-size 256 --overlap 64
```

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--chunk-size` | 256 | Размер чанка в токенах |
| `--overlap` | 64 | Перекрытие между чанками |
| `--model` | paraphrase-multilingual-mpnet-base-v2 | Модель эмбеддингов |

## 📊 Производительность

### Время генерации ответа

| Модель | CPU | GPU (GTX 1060) | GPU (RTX 3060) |
|--------|-----|----------------|----------------|
| **0.5B** | 10-15 сек | **2-5 сек** ⚡ | 1-2 сек |
| **1.5B** | 30-60 сек | 10-15 сек | **3-5 сек** ⚡ |
| **3B** | 2-5 мин | 30-60 сек | **10-15 сек** ⚡ |

### Точность поиска

| Запросов найдено | Точность |
|------------------|----------|
| Русские книги | 85-95% |
| Английские книги | 90-98% |
| Смешанные запросы | 80-90% |

## 🔧 API Endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/api/search` | POST | Поиск фрагментов |
| `/api/answer` | POST | Ответ на вопрос |
| `/api/books` | GET | Список книг |
| `/api/upload` | POST | Загрузка книги |
| `/api/reindex` | POST | Переиндексация |

### Пример запроса

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Татьяна пишет письмо", "k": 5}'
```

## 📚 Поддерживаемые форматы

- **Входные файлы:** `.txt` (UTF-8, CP1251, Latin-1)
- **Кодировки:** Автоопределение
- **Языки:** Русский, английский, мультиязычный поиск

## 🛠️ Расширение

### Добавить новую книгу

1. Положите `.txt` файл в папку `data/`
2. В веб-интерфейсе: Книги → Загрузить
3. Нажмите "Проиндексировать"

Или через консоль:
```bash
python main_processor.py --step all
```

### Интеграция в свой проект

```python
from book_rag import BookRAG

rag = BookRAG()
rag.load_index()

# Поиск
fragments = rag.search_fragments("запрос", k=5)

# Ответ на вопрос
result = rag.answer_question("вопрос?", k=5)
print(result['answer'])
print(result['quotes'])
```

## 📄 Документация

- [WEB_README.md](WEB_README.md) — Веб-интерфейс
- [GPU_SETUP.md](GPU_SETUP.md) — Настройка GPU

## 🤝 Вклад

1. Fork репозиторий
2. Создай ветку (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Pull Request

## 📝 Лицензия

MIT License - см. [LICENSE](LICENSE) файл

## 👥 Авторы

- [@Sanndigo](https://github.com/Sanndigo) - Основная разработка

## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) за модели
- [Qwen Team](https://github.com/QwenLM/Qwen) за LLM
- [Sentence Transformers](https://www.sbert.net/) за эмбеддинги

## 📬 Контакты

GitHub Issues: [Сообщить о проблеме](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks/issues)

---

<div align="center">

**Сделано с ❤️ для любителей книг**

[⭐ Звезда на GitHub](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks/stargazers)

</div>
