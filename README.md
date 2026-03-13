# 📚 RAG - Поиск по книгам с AI-ответами

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial](https://img.shields.io/badge/license-NC--BY-red.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.3.4-green.svg)](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks)

Интеллектуальная система поиска и ответов на вопросы по текстовым документам с использованием RAG (Retrieval-Augmented Generation) и **YandexGPT API**.

## ✨ Возможности

### 🔍 Умный поиск
- Мгновенный поиск по всем загруженным книгам
- Показывает 5 наиболее релевантных фрагментов
- **Отображает главы, строфы, части** если найдены
- Указывает книгу и процент схожести

### ❓ Ответы на вопросы с YandexGPT
- AI генерирует развернутые ответы на русском языке
- **3000 запросов в месяц бесплатно**
- Работает через облако — не нужен мощный GPU
- Показывает цитаты из книг для подтверждения

### 📖 Управление библиотекой
- Веб-интерфейс с современным дизайном
- Загрузка TXT файлов через drag-and-drop
- Автоматическая индексация новых книг
- Поддержка русских и английских текстов

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка YandexGPT API

**Получи ключи:**
1. Зайди на https://cloud.yandex.ru/
2. Создай сервисный аккаунт с ролью `ai.languageModels.user`
3. Создай API-ключ
4. Скопируй Folder ID

**Настрой .env:**
```bash
# Создай файл .env в корне проекта
echo YANDEXGPT_API_KEY=AQVN...твой_ключ > .env
echo YANDEXGPT_FOLDER_ID=b1g...твой_id >> .env
```

### 3. Запуск веб-интерфейса

```bash
python web_app.py
```

Откройте в браузере: **http://localhost:5000**

### 4. Консольный режим

```bash
python book_rag.py --interactive
```

## 📋 Требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **RAM** | 4 GB | 8 GB |
| **Место** | 2 GB | 6 GB |
| **GPU** | Не требуется | Любой (опционально) |

### Для YandexGPT API:
- Интернет соединение
- API ключи Yandex Cloud
- **Бесплатно:** 3000 запросов/мес

## 📁 Структура проекта

```
TechnoStr/
├── web_app.py              # Flask веб-сервер
├── book_rag.py             # RAG движок с YandexGPT
├── embedding_processor.py  # Векторный поиск (FAISS)
├── text_processor.py       # Обработка текста
├── main_processor.py       # Обработка книг
├── templates/
│   └── index.html          # Веб-интерфейс
├── data/                   # TXT книги
├── processed/              # Обработанные чанки (игнорируется git)
└── embeddings/             # Векторный индекс
```

## 🎯 Примеры использования

### Поиск фрагмента

**Запрос:** "где Татьяна пишет письмо Онегину"

**Результат:**
```
📚 Найдено фрагментов: 5
----------------------------------------------------------------------
1. 📖 Евгений Онегин
   📍 Строфа XXV
   📊 Схожесть: 85.3%
   📝 Но получив посланье Тани...
```

### Ответ на вопрос

**Вопрос:** "Кто написал Шинель?"

**Ответ:**
```
======================================================================
 💬 ОТВЕТ
======================================================================
  Согласно тексту из книги "Шинель", автором является
  Николай Васильевич Гоголь.

======================================================================
 📑 ИСТОЧНИКИ (цитаты из книг)
======================================================================
  [1] Шинель (фрагмент #1)
      «Николай Васильевич Гоголь Шинель...»
```

## 🔧 API Endpoints (для разработчиков)

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/api/search` | POST | Поиск фрагментов |
| `/api/answer` | POST | Ответ на вопрос |
| `/api/books` | GET | Список книг |
| `/api/upload` | POST | Загрузка книги |
| `/api/reindex` | POST | Переиндексация |

### Пример запроса через curl:

```bash
# Поиск
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Татьяна пишет письмо", "k": 5}'

# Ответ на вопрос
curl -X POST http://localhost:5000/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Кто написал Шинель?", "k": 5}'
```

## 💰 Тарифы YandexGPT

| Тариф | Лимит | Цена |
|-------|-------|------|
| **Free** | 3000 запросов/мес | 0 ₽ |
| **Pay-as-you-go** | По факту | ~0.5 ₽ за 1K токенов |

Для личного использования хватит бесплатного лимита!

## ⚙️ Настройка

### Переменные окружения (.env)

```bash
# YandexGPT API
YANDEXGPT_API_KEY=AQVN...твой_api_key
YANDEXGPT_FOLDER_ID=b1g...твой_folder_id
```

### Параметры обработки книг

```bash
# Обработка с кастомными параметрами
python main_processor.py --chunk-size 256 --overlap 64
```

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--chunk-size` | 256 | Размер чанка в токенах |
| `--overlap` | 64 | Перекрытие между чанками |

## 🛠️ Расширение

### Добавить новую книгу

1. Положи `.txt` файл в папку `data/`
2. В веб-интерфейсе: Книги → Загрузить
3. Нажми "Проиндексировать"

Или через консоль:
```bash
python main_processor.py --step all
```

### Интеграция в свой проект

```python
from book_rag import BookRAG

# Инициализация с YandexGPT
rag = BookRAG(
    llm_model_name='YANDEXGPT',
    yandexgpt_api_key='AQVN...',
    yandexgpt_folder_id='b1g...'
)
rag.load_index()

# Поиск
fragments = rag.search_fragments("запрос", k=5)

# Ответ на вопрос
result = rag.answer_question("вопрос?", k=5)
print(result['answer'])
print(result['quotes'])
```

## 📊 Производительность

### Время ответа

| Операция | Время |
|----------|-------|
| Поиск фрагментов | < 1 сек |
| Ответ YandexGPT | 2-5 сек |
| Загрузка книги | 1-2 мин |

### Точность поиска

| Тип запроса | Точность |
|-------------|----------|
| Русские книги | 85-95% |
| Английские книги | 90-98% |
| Смешанные запросы | 80-90% |

## 🐛 Решение проблем

### "YandexGPT API error: 401"
- Проверь API ключ в `.env`
- Убедись что сервисный аккаунт активен
- Проверь роль `ai.languageModels.user`

### "YandexGPT API error: 429"
- Превышен лимит запросов (3000/мес)
- Подожди следующего месяца

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install openai
```

### Книги не индексируются
- Проверь кодировку файлов (UTF-8 или CP1251)
- Удали папку `processed/` и запусти заново

## 📝 Лицензия

**Non-Commercial License (CC BY-NC 4.0 Compatible)**

**Copyright Holder:** Serafim Grekov (Sanndigo)

- ✅ **Разрешено:** личное использование, образование, исследования
- ❌ **Запрещено:** коммерческое использование, продажа, монетизация
- 📋 **Требования:** указание автора, сохранение лицензии

Действует на территории: **Российская Федерация** и **США**

См. полный текст в файле [LICENSE](LICENSE)

## 👥 Авторы

- [@Sanndigo](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks)

## 🤝 Вклад

1. Fork репозиторий
2. Создай ветку (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Pull Request

## 📬 Контакты

GitHub Issues: [Сообщить о проблеме](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks/issues)

---

<div align="center">

**Сделано с ❤️ для любителей книг**

[⭐ Звезда на GitHub](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks/stargazers)

</div>
