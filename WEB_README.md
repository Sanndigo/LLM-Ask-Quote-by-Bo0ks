# 🚀 Запуск веб-интерфейса RAG-системы

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск веб-сервера

```bash
python web_app.py
```

После запуска откройте в браузере:
**http://localhost:5000**

## Функционал веб-интерфейса

### 🔍 Поиск фрагментов
- Введите запрос (например: "где Татьяна пишет письмо")
- Система найдет 5 наиболее релевантных фрагментов
- Для каждого фрагмента указывается:
  - 📖 Книга
  - 📍 Позиция (глава/строфа + номер фрагмента)
  - 📊 Процент схожести
  - 📝 Текст фрагмента

### ❓ Ответ на вопрос
- Введите вопрос (например: "Что писала Татьяна Онегину?")
- Система сгенерирует ответ на основе контекста
- Покажет цитаты из книг с источниками

### 📖 Книги
- **Загрузка**: Перетащите TXT файл или кликните для выбора
- **Библиотека**: Просмотр всех загруженных книг
- **Индексация**: После загрузки новой книги нажмите "Проиндексировать"

## Структура проекта

```
TechnoStr/
├── web_app.py           # Flask веб-приложение
├── book_rag.py          # RAG-система (поиск + ответы)
├── embedding_processor.py # Обработка эмбеддингов
├── text_processor.py    # Обработка текста
├── main_processor.py    # Основной процессор
├── templates/
│   └── index.html       # Веб-интерфейс
├── data/                # TXT книги
├── processed/           # Обработанные чанки
└── embeddings/          # FAISS индекс
```

## API Endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/api/search` | POST | Поиск фрагментов |
| `/api/answer` | POST | Ответ на вопрос |
| `/api/books` | GET | Список книг |
| `/api/upload` | POST | Загрузка книги |
| `/api/reindex` | POST | Переиндексация |
| `/api/indexing-status` | GET | Статус индексации |

## Примеры запросов

### Поиск фрагментов
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Татьяна пишет письмо", "k": 5}'
```

### Ответ на вопрос
```bash
curl -X POST http://localhost:5000/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Кто главный герой Шинели?", "k": 5}'
```

### Загрузка книги
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@my_book.txt"
```

## Требования

- Python 3.10+
- **8GB+ RAM** (для Qwen2.5-3B, минимум 4GB для 1.5B)
- **6GB+** свободного места для модели
- GPU опционально (ускоряет генерацию в 5-10 раз)

## Поддерживаемые модели

- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2`
- **LLM (по умолчанию)**: `Qwen/Qwen2.5-3B-Instruct` ⭐
- **LLM (fallback)**: `Qwen/Qwen2.5-1.5B-Instruct`

### Альтернативные модели:

```python
# В book_rag.py измени:
llm_model_name: str = 'Qwen/Qwen2.5-3B-Instruct'  # 3B - отличный баланс
llm_model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'  # 1.5B - быстрее, меньше памяти
llm_model_name: str = 'microsoft/Phi-3-mini-4k-instruct'  # 3.8B - хорош для английского
```
