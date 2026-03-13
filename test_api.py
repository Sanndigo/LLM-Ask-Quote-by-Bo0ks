from book_rag import GigaChatAPI

print("Инициализация GigaChatAPI...")
api = GigaChatAPI()

print(f"Token получен: {api.token is not None}")
if api.token:
    print(f"Токен: {api.token[:50]}...")
    
    # Тестовый запрос
    print("\nТестовый запрос...")
    answer = api.generate("Привет! Как тебя зовут?", max_tokens=50)
    print(f"Ответ: {answer}")
else:
    print("❌ Токен не получен!")
