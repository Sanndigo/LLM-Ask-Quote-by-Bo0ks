#!/usr/bin/env python3
"""
RAG-система для работы с книгами
Поиск фрагментов и ответы на вопросы с цитатами
Использует Qwen API через OpenRouter
"""
import os
import sys
from typing import List, Dict, Optional
from embedding_processor import EmbeddingProcessor
from text_processor import TextProcessor
import logging
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Загружаем .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MistralAPI:
    """Mistral AI API - бесплатно 30 дней trial
    https://console.mistral.ai/
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        self.url = 'https://api.mistral.ai/v1/chat/completions'
        self.model = 'mistral-small-latest'  # Бесплатная модель
        
        if self.api_key:
            logger.info(f"✅ Mistral API: {self.api_key[:8]}...")
        else:
            logger.error("❌ MISTRAL_API_KEY не найден!")
            logger.info("   Получи на https://console.mistral.ai/api-keys")
    
    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        if not self.api_key:
            return "Нет API ключа. Получи на https://console.mistral.ai/"
        
        import requests
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens
        }
        
        try:
            r = requests.post(self.url, headers=headers, json=data, timeout=60)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            else:
                return f"Ошибка Mistral API: {r.status_code} - {r.text[:100]}"
        except Exception as e:
            return f"Ошибка: {e}"


class BookRAG:
    def __init__(self, index_path='embeddings/faiss_index.bin',
                 id_map_path='embeddings/faiss_index.bin_id_map.pkl',
                 model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 api_key=None):
        self.embedding_processor = EmbeddingProcessor(model_name)
        self.embedding_processor.load_index(index_path, id_map_path)
        self.mistral = MistralAPI(api_key)
        logger.info("✅ RAG система готова")
    
    def search(self, query: str, k=5) -> List[Dict]:
        results = self.embedding_processor.search(query, k)
        fragments = []
        for chunk_id, dist in results:
            content = self.embedding_processor.get_chunk_content(chunk_id)
            if content:
                # Определяем источник
                source = "Неизвестно"
                chapter = ""
                if chunk_id in self.embedding_processor.chunk_paths:
                    path = self.embedding_processor.chunk_paths[chunk_id]
                    filename = os.path.basename(path)
                    # Извлекаем имя книги
                    parts = filename.replace('.txt', '').split('_chunk_')
                    if len(parts) == 2:
                        book_name = parts[0]
                        if 'EvgeniyOnegin' in book_name:
                            source = "Евгений Онегин"
                        elif 'Shinell' in book_name:
                            source = "Шинель"
                        elif 'VlastelinKolec' in book_name:
                            source = "Властелин Колец"
                        else:
                            source = book_name
                
                fragments.append({
                    'id': chunk_id,
                    'similarity': 1.0 - dist,
                    'content': content[:400],
                    'full': content,
                    'source': source,
                    'chapter': chapter
                })
        return sorted(fragments, key=lambda x: x['similarity'], reverse=True)
    
    def answer(self, question: str, k=5) -> Dict:
        fragments = self.search(question, k)
        if not fragments:
            return {'answer': 'Ничего не найдено', 'quotes': [], 'found': False}

        # Определяем основную книгу по фрагментам
        book_counts = {}
        for f in fragments:
            book = f.get('source', 'Книга')
            book_counts[book] = book_counts.get(book, 0) + 1
        
        book_name = max(book_counts, key=book_counts.get) if book_counts else 'Книга'

        # Промпт только с названием книги
        prompt = f"""Пользователь спрашивает о книге: "{book_name}"

Вопрос: {question}

Дай развернутый ответ на русском языке, основываясь на содержании книги "{book_name}".
Упоминай главы, строфы, части если они есть.
Если не знаешь ответа — так и скажи.
Если данная тебе книга не соотвествует вопросу - ответь "К сожалению у меня нет информации по данному вопросу, возможно вам стоит загрузить соотвутствующую книгу в проагрмму", больше ничего не отвечай.
Не используй форматирование (bold, italic)."""

        answer = self.mistral.generate(prompt, max_tokens=1500)
        return {
            'answer': answer,
            'quotes': [{'text': f['content'], 'source': f['source'], 'similarity': f'{f["similarity"]:.0%}'} for f in fragments[:3]],
            'found': True
        }


if __name__ == '__main__':
    rag = BookRAG()
    result = rag.answer('Кто написал Шинель?')
    print(result['answer'])
