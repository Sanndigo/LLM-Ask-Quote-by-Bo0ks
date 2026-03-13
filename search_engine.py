#!/usr/bin/env python3
"""
Поисковая система для поиска фрагментов в загруженных текстах
"""
import os
import sys
from typing import List, Tuple, Dict
from embedding_processor import EmbeddingProcessor
from text_processor import TextProcessor
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchEngine:
    """Поисковая система для поиска фрагментов в текстах"""
    
    def __init__(self, index_path: str = 'embeddings/faiss_index.bin',
                 id_map_path: str = 'embeddings/faiss_index.bin_id_map.pkl',
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """
        Инициализация поисковой системы
        
        Args:
            index_path: Путь к файлу индекса FAISS
            id_map_path: Путь к файлу ID-карты
            model_name: Название модели для создания эмбеддингов
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.model_name = model_name
        self.embedding_processor = None
        self.loaded = False
    
    def load_index(self):
        """Загрузка индекса из файлов"""
        try:
            self.embedding_processor = EmbeddingProcessor(self.model_name)
            self.embedding_processor.load_index(self.index_path, self.id_map_path)
            self.loaded = True
            logger.info("Индекс успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Поиск ближайших фрагментов к запросу
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список кортежей (ID фрагмента, расстояние)
        """
        if not self.loaded:
            self.load_index()
        
        # Поиск в индексе
        results = self.embedding_processor.search(query, k)
        return results
    
    def search_with_context(self, query: str, k: int = 5) -> List[Dict]:
        """
        Поиск с контекстом - возвращает фрагменты с информацией о результатах

        Args:
            query: Поисковый запрос
            k: Количество результатов

        Returns:
            Список словарей с информацией о результатах
        """
        if not self.loaded:
            self.load_index()

        # Поиск в индексе
        results = self.embedding_processor.search(query, k)

        # Возвращаем результаты с информацией о расстоянии и контентом
        search_results = []
        query_lower = query.lower()
        
        for chunk_id, distance in results:
            content = self.embedding_processor.get_chunk_content(chunk_id)
            if content:
                # Ищем место где встречается запрос или близкие слова
                content_lower = content.lower()
                start_idx = 0
                
                # Пробуем найти часть запроса в контенте
                for word in query_lower.split()[:3]:  # Берем первые 3 слова
                    if len(word) > 3:
                        idx = content_lower.find(word)
                        if idx != -1:
                            # Показываем контекст вокруг найденного слова
                            start_idx = max(0, idx - 50)
                            break
                
                # Если не нашли, показываем с начала, но пропускаем шапку библиотеки
                if start_idx == 0 and 'royallibru' in content_lower[:200]:
                    # Пропускаем шапку до первого нормального текста
                    for marker in ['\n\n', '  ']:
                        idx = content.find(marker)
                        if idx > 50:
                            start_idx = idx
                            break
                
                # Показываем фрагмент ~300 символов
                end_idx = min(len(content), start_idx + 300)
                display_content = content[start_idx:end_idx]
                if start_idx > 0:
                    display_content = "..." + display_content
                if end_idx < len(content):
                    display_content = display_content + "..."
            else:
                display_content = f'[Контент недоступен для чанка {chunk_id}]'
            
            result = {
                'chunk_id': chunk_id,
                'distance': distance,
                'similarity': 1.0 - distance,
                'content': display_content
            }
            search_results.append(result)

        # Сортируем по схожести (убывание)
        search_results.sort(key=lambda x: x['similarity'], reverse=True)

        return search_results

def interactive_search():
    """Интерактивный режим поиска"""
    print("=== Интерактивный поиск по текстам ===")
    print("Введите 'quit' для выхода")
    
    # Инициализация поисковой системы
    search_engine = SearchEngine()
    
    try:
        search_engine.load_index()
        print("Поисковая система готова к работе")
        
        while True:
            try:
                query = input("\nВведите поисковый запрос: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Выход из поисковой системы")
                    break
                    
                if not query:
                    print("Пожалуйста, введите запрос")
                    continue
                
                print(f"\nПоиск по запросу: '{query}'")
                
                # Выполняем поиск
                results = search_engine.search_with_context(query, k=3)
                
                if not results:
                    print("Ничего не найдено")
                    continue
                
                print(f"\nНайдено {len(results)} результатов:")
                print("-" * 50)

                for i, result in enumerate(results, 1):
                    print(f"{i}. ID: {result['chunk_id']}")
                    print(f"   Схожесть: {result['similarity']:.4f}")
                    print(f"   Расстояние: {result['distance']:.4f}")
                    print(f"   Контент: {result['content'][:200]}...")
                    print()
                    
            except KeyboardInterrupt:
                print("\n\nВыход из поисковой системы")
                break
            except EOFError:
                print("\n\nВыход из поисковой системы")
                break
                
    except Exception as e:
        logger.error(f"Ошибка в интерактивном режиме: {str(e)}")
        print(f"Ошибка: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Можно запустить интерактивный режим
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_search()
    else:
        # Пример программного использования
        search_engine = SearchEngine()
        try:
            search_engine.load_index()
            print("Поисковая система готова")
            
            # Пример поиска
            results = search_engine.search_with_context("natural language processing", k=2)
            print("Результаты поиска:", results)
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")