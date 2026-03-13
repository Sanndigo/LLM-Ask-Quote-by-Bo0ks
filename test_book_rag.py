from book_rag import BookRAG

print("Инициализация BookRAG...")
try:
    rag = BookRAG()
    print("✅ BookRAG готов!")
    
    # Тест поиска
    print("\nТест поиска...")
    results = rag.search("Татьяна Онегин", k=3)
    print(f"Найдено фрагментов: {len(results)}")
    if results:
        print(f"Первый фрагмент: {results[0]['similarity']:.0%} схожесть")
    
    print("\n✅ Все работает!")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
