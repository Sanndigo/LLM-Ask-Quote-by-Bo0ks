# 🔌 Настройка GigaChat API

## Что такое GigaChat?

GigaChat — это мощная русскоязычная AI модель от Сбера. Работает через API, не требует мощного железа!

### Преимущества:
- ✅ **Не нужен мощный GPU** — всё работает в облаке
- ✅ **Отличный русский язык** — лучше любых локальных моделей
- ✅ **Быстрые ответы** — 2-5 секунд
- ✅ **Качественные ответы** — не выдумывает факты
- ✅ **Бесплатный лимит** — 1000 запросов в месяц бесплатно

---

## 📋 Пошаговая настройка

### 1. Регистрация на SberDevices

1. Зайди на https://developers.sber.ru/
2. Нажми **"Войти"** → войди через Сбер ID
3. Перейди в **"Профиль"** → **"Мои проекты"**
4. Нажми **"Создать проект"**
5. Дай название (например "RAG Books")

### 2. Получение ключей

1. В проекте перейди во вкладку **"GigaChat"**
2. Нажми **"Создать новый ключ"**
3. Скопируй:
   - **Client ID** (выглядит как `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
   - **Client Secret** (длинная строка)

⚠️ **Сохрани Client Secret** — его можно увидеть только один раз!

### 3. Настройка в проекте

Создай файл `.env` в корне проекта:

```bash
# Файл: .env
GIGACHAT_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
GIGACHAT_CLIENT_SECRET=твоя_секретная_строка
```

Или передай ключи напрямую в коде:

```python
# book_rag.py, строка ~115
rag = BookRAG(
    llm_model_name='GIGACHAT',
    gigachat_client_id='xxx-xxx-xxx',
    gigachat_client_secret='secret'
)
```

### 4. Проверка работы

```bash
python book_rag.py --interactive
```

Введи вопрос — должен прийти ответ от GigaChat!

---

## 💰 Тарифы

| Тариф | Лимит | Цена |
|-------|-------|------|
| **Бесплатный** | 1000 запросов/мес | 0 ₽ |
| **Базовый** | 10 000 запросов/мес | 990 ₽/мес |
| **Продвинутый** | 100 000 запросов/мес | 4990 ₽/мес |

Для личного использования хватит бесплатного!

---

## 🔧 Использование в веб-интерфейсе

В `web_app.py` добавь ключи при инициализации:

```python
rag = BookRAG(
    llm_model_name='GIGACHAT',
    gigachat_client_id=os.getenv('GIGACHAT_CLIENT_ID'),
    gigachat_client_secret=os.getenv('GIGACHAT_CLIENT_SECRET')
)
```

Или через переменные окружения (рекомендуется):

```bash
# Windows PowerShell
$env:GIGACHAT_CLIENT_ID="xxx"
$env:GIGACHAT_CLIENT_SECRET="xxx"
python web_app.py

# Linux/Mac
export GIGACHAT_CLIENT_ID="xxx"
export GIGACHAT_CLIENT_SECRET="xxx"
python web_app.py
```

---

## 🐛 Решение проблем

### "GigaChat auth error: 401"
- Проверь Client ID и Client Secret
- Убедись что проект активен
- Проверь срок действия ключей

### "GigaChat API error: 429"
- Превышен лимит запросов
- Подожди следующего месяца или обнови тариф

### "GigaChat API error: 400"
- Неправильный формат запроса
- Проверь версию API (v2)

---

## 📊 Сравнение с локальными моделями

| Модель | Скорость | Качество | Русский | Железо |
|--------|----------|----------|---------|--------|
| **GigaChat API** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Любое |
| Qwen2.5-3B | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 6GB VRAM |
| Qwen2.5-0.5B | ⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | 2GB VRAM |
| Phi-3-mini | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 3GB VRAM |

**Вывод:** GigaChat лучше по всем параметрам если есть интернет!

---

## 🔗 Полезные ссылки

- [Документация GigaChat](https://developers.sber.ru/docs/gigachat)
- [Примеры кода](https://github.com/sberdevices/gigachat)
- [Тарифы](https://developers.sber.ru/tariffs)
