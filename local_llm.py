#!/usr/bin/env python3
"""
Локальная LLM-модель для работы с контекстом загруженных TXT файлов
"""
import os
import sys
from typing import List, Dict, Optional
from embedding_processor import EmbeddingProcessor
from text_processor import TextProcessor
import logging
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalLLM:
    """Локальная LLM-модель для работы с контекстом документов"""
    
    def __init__(self, index_path: str = 'embeddings/faiss_index.bin',
                 id_map_path: str = 'embeddings/faiss_index.bin_id_map.pkl',
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_model_name: str = 'distilgpt2'):
        """
        Инициализация локальной LLM
        
        Args:
            index_path: Путь к файлу индекса FAISS
            id_map_path: Путь к файлу ID-карты
            model_name: Название модели для создания эмбеддингов
            llm_model_name: Название модели LLM для генерации ответов
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.embedding_processor = None
        self.text_processor = TextProcessor()
        self.loaded = False
        self.llm_model = None
        self.tokenizer = None
        
        # Инициализация модели для создания эмбеддингов
        self.embedder = SentenceTransformer(model_name)
        
        # Загрузка LLM модели (если доступна)
        self._load_llm_model()
    
    def _load_llm_model(self):
        """Загрузка LLM модели"""
        try:
            # Попробуем загрузить легковесную модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загрузка модели с использованием torch_dtype=torch.float16 для экономии памяти
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Переводим модель в режим оценки
            self.llm_model.eval()
            
            logger.info(f"LLM модель {self.llm_model_name} загружена успешно")
        except Exception as e:
            logger.warning(f"Не удалось загрузить LLM модель {self.llm_model_name}: {str(e)}")
            logger.info("Будет использоваться демонстрационный режим без генерации ответов")
    
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
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """
        Получение контекста из документов по запросу
        
        Args:
            query: Поисковый запрос
            k: Количество возвращаемых фрагментов
            
        Returns:
            Список словарей с фрагментами и метриками
        """
        if not self.loaded:
            self.load_index()
        
        # Поиск ближайших эмбеддингов
        results = self.embedding_processor.search(query, k)
        
        # Подготовка контекста (в реальной реализации здесь будет загрузка
        # самих фрагментов из файлов)
        context_fragments = []
        for chunk_id, distance in results:
            fragment_info = {
                'chunk_id': chunk_id,
                'distance': distance,
                'similarity': 1.0 - distance,
                'content': f'Фрагмент {chunk_id} - контекст по запросу: "{query}"'
            }
            context_fragments.append(fragment_info)
        
        return context_fragments
    
    def generate_response(self, query: str, context_fragments: List[Dict]) -> str:
        """
        Генерация ответа на основе контекста
        
        Args:
            query: Вопрос пользователя
            context_fragments: Фрагменты контекста
            
        Returns:
            Сгенерированный ответ
        """
        # Если LLM модель не загружена, используем демонстрационный режим
        if self.llm_model is None or self.tokenizer is None:
            context_summary = "\n".join([
                f"- {fragment['content'][:100]}..."
                for fragment in context_fragments[:2]
            ])
            
            response = f"""
Ответ на запрос: "{query}"

Контекст из документов:
{context_summary}

Важно: Локальная модель не загружена. В реальной реализации здесь будет
использоваться легковесная LLM (например, DistilGPT-2 или другая)
для генерации точного ответа на основе предоставленного контекста.
            """.strip()
            return response
        
        # В реальной реализации здесь будет использование LLM для генерации ответа
        # Для демонстрации просто формируем ответ из контекста
        context_summary = "\n".join([
            f"- {fragment['content'][:100]}..."
            for fragment in context_fragments[:2]
        ])
        
        response = f"""
Ответ на запрос: "{query}"

Контекст из документов:
{context_summary}

Это демонстрационный ответ. В реальной реализации здесь будет
использоваться локальная LLM для генерации точного ответа на основе
предоставленного контекста.
        """.strip()
        
        return response
    
    def chat(self, query: str) -> str:
        """
        Интерактивный чат с локальной моделью
        
        Args:
            query: Вопрос пользователя
            
        Returns:
            Ответ модели
        """
        if not self.loaded:
            self.load_index()
        
        # Получение контекста
        context_fragments = self.retrieve_context(query, k=3)
        
        # Генерация ответа
        response = self.generate_response(query, context_fragments)
        
        return response

def interactive_local_chat():
    """Интерактивный режим чата с локальной моделью"""
    print("=== Локальная LLM для работы с документами ===")
    print("Введите 'quit' для выхода")
    print("Запросы будут обрабатываться в контексте загруженных документов")
    
    # Инициализация локальной LLM
    local_llm = LocalLLM()
    
    try:
        local_llm.load_index()
        print("Локальная модель готова к работе")
        print("Все запросы будут анализироваться по контексту документов")
        
        while True:
            try:
                query = input("\nВаш вопрос: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Выход из локальной LLM")
                    break
                    
                if not query:
                    print("Пожалуйста, введите вопрос")
                    continue
                
                print(f"\nОбработка запроса: '{query}'")
                print("Поиск контекста...")
                
                # Получение ответа
                response = local_llm.chat(query)
                print(f"\nОтвет:")
                print(response)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nВыход из локальной LLM")
                break
            except EOFError:
                print("\n\nВыход из локальной LLM")
                break
                
    except Exception as e:
        logger.error(f"Ошибка в интерактивном режиме: {str(e)}")
        print(f"Ошибка: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Можно запустить интерактивный режим
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_local_chat()
    else:
        # Пример программного использования
        local_llm = LocalLLM()
        try:
            local_llm.load_index()
            print("Локальная LLM готова")
            
            # Пример запроса
            response = local_llm.chat("Что такое Natural Language Processing?")
            print("Ответ на запрос:", response[:200] + "...")
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")