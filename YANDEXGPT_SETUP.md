# 🔌 Настройка YandexGPT API (OpenAI SDK)

## Что такое YandexGPT?

YandexGPT — это мощная русскоязычная AI модель от Яндекса. Работает через API в формате OpenAI-compatible!

### Преимущества:
- ✅ **OpenAI-compatible API** — работает через стандартный OpenAI SDK
- ✅ **Не нужен мощный GPU** — всё работает в облаке
- ✅ **Отличный русский язык** — родная модель для русского
- ✅ **Быстрые ответы** — 2-5 секунд
- ✅ **Бесплатный лимит** — 3000 запросов/мес бесплатно

---

## 📋 Пошаговая настройка

### 1. Регистрация в Yandex Cloud

1. Зайди на https://cloud.yandex.ru/
2. Нажми **"Войти"** → войди через Яндекс ID
3. Создай новый платежный профиль (требуется для API)

### 2. Создание сервисного аккаунта

1. Перейди в **Консоль разработки** → https://console.cloud.yandex.ru/
2. Выбери **Сервисные аккаунты** в левом меню
3. Нажми **"Создать сервисный аккаунт"**
4. Дай название (например "rag-books")
5. Выбери роль **`ai.languageModels.user`** (доступ к YandexGPT)
6. Нажми **"Создать"**

### 3. Создание API ключа

1. В созданном сервисном аккаунте нажми **"Создать новый ключ"**
2. Выбери тип ключа **API-ключ**
3. Скопируй ключ (выглядит как `AQVN...` длинная строка)
4. Сохрани в надежное место!

⚠️ **API ключ показывается только один раз!**

### 4. Получение Folder ID

1. В консоли перейди в **Облако** (верхняя панель)
2. Скопируй **ID облака** (выглядит как `b1gxxxxxxxxxxxxxx`)
3. Это твой Folder ID

### 5. Установка зависимостей

```bash
pip install openai
```

### 6. Настройка в проекте

Создай файл `.env` в корне проекта:

```bash
# Файл: .env
YANDEXGPT_API_KEY=AQVN...твой_api_key...
YANDEXGPT_FOLDER_ID=b1g...твой_folder_id...
```

Или передай ключи напрямую в коде:

```python
# book_rag.py, строка ~106
rag = BookRAG(
    llm_model_name='YANDEXGPT',
    yandexgpt_api_key='AQVN...',
    yandexgpt_folder_id='b1g...'
)
```

### 7. Проверка работы

```bash
python book_rag.py --interactive
```

Введи вопрос — должен прийти ответ от YandexGPT!

---

## 💰 Тарифы

| Тариф | Лимит | Цена |
|-------|-------|------|
| **Free** | 3000 запросов/мес | 0 ₽ |
| **Pay-as-you-go** | По факту | ~0.5 ₽ за 1000 токенов |

Для личного использования хватит бесплатного!

---

## 🔧 Пример кода

```python
import openai

YANDEX_CLOUD_FOLDER = "b1g99cd542d9k55lqihc"
YANDEX_CLOUD_API_KEY = "AQVN..."
YANDEX_CLOUD_MODEL = "yandexgpt/latest"

client = openai.OpenAI(
  api_key=YANDEX_CLOUD_API_KEY,
  base_url="https://ai.api.cloud.yandex.net/v1",
  project=YANDEX_CLOUD_FOLDER
)

response = client.responses.create(
  model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
  temperature=0.3,
  input="Придумай заголовок для статьи о будущем ИИ",
  max_output_tokens=500
)

print(response.output_text)
```

---

## 🐛 Решение проблем

### "OpenAI API error: 401"
- Проверь API ключ
- Убедись что сервисный аккаунт активен
- Проверь роль `ai.languageModels.user`

### "OpenAI API error: 403"
- Нет доступа к YandexGPT
- Проверь роль сервисного аккаунта
- Убедись что Folder ID правильный

### "OpenAI API error: 429"
- Превышен лимит запросов
- Подожди следующего месяца или обнови тариф

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install openai
```

---

## 📊 Сравнение с локальными моделями

| Модель | Скорость | Качество | Русский | Железо |
|--------|----------|----------|---------|--------|
| **YandexGPT API** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐⭐ | Любое |
| Qwen2.5-3B | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 6GB VRAM |
| Qwen2.5-0.5B | ⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | 2GB VRAM |
| Phi-3-mini | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 3GB VRAM |

**Вывод:** YandexGPT лучше по всем параметрам если есть интернет!

---

## 🔗 Полезные ссылки

- [Документация YandexGPT](https://yandex.cloud/ru/docs/foundation-models/concepts/yandexgpt/)
- [OpenAI-compatible API](https://yandex.cloud/ru/docs/foundation-models/api-ref/openai/)
- [Примеры кода](https://github.com/yandex-cloud/examples)
- [Тарифы](https://yandex.cloud/ru/pricing?services=ai)
