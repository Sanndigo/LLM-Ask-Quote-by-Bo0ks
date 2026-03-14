#!/usr/bin/env python3
"""
Семантический чанкинг текста
Разбивает текст ТОЛЬКО по предложениям (от точки до точки)
Группирует предложения по семантической схожести
"""
import os
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticTextProcessor:
    """
    Процессор для семантического чанкинга
    
    Правила:
    1. Разбиваем текст ТОЛЬКО по предложениям (от точки до точки)
    2. Считаем эмбеддинги для каждого предложения
    3. Объединяем предложения в чанки по схожести
    4. НИКОГДА не обрываем слова или предложения
    """
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        similarity_threshold: float = 0.45,  # Порог схожести
        min_sentences: int = 2,  # Минимум предложений в чанке
        max_sentences: int = 10,  # Максимум предложений в чанке
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        
        logger.info(f"Загрузка модели: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        logger.info("✅ Модель загружена")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения
        ТОЛЬКО по точкам, восклицательным и вопросительным знакам
        """
        # Очищаем от лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Разбиваем по предложениям
        # Используем regex для сохранения разделителей
        parts = re.split(r'([.!?]+\s*)', text)
        
        sentences = []
        current = ''
        
        for part in parts:
            current += part
            # Если это разделитель (. ! ?) - завершаем предложение
            if re.match(r'^[.!?]+\s*$', part):
                sentence = current.strip()
                if len(sentence) > 10:  # Минимальная длина
                    sentences.append(sentence)
                current = ''
        
        # Добавляем последнее если есть
        if current.strip() and len(current.strip()) > 10:
            sentences.append(current.strip())
        
        logger.debug(f"Разбито на {len(sentences)} предложений")
        return sentences
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """
        Создание семантических чанков
        
        Алгоритм:
        1. Разбиваем на предложения (от точки до точки)
        2. Считаем эмбеддинги для каждого
        3. Находим где схожесть падает ниже порога
        4. Объединяем в чанки
        """
        # 1. Разбиваем на предложения
        sentences = self.split_into_sentences(text)
        
        if len(sentences) == 0:
            return []
        
        if len(sentences) == 1:
            return sentences
        
        logger.debug(f"Обработка {len(sentences)} предложений")
        
        # 2. Считаем эмбеддинги
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
        
        # 3. Вычисляем схожесть между соседними
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # 4. Находим границы чанков
        boundaries = self._find_boundaries(similarities)
        
        # 5. Формируем чанки (ТОЛЬКО целые предложения!)
        chunks = self._build_chunks(sentences, boundaries)
        
        logger.info(f"Создано {len(chunks)} семантических чанков")
        return chunks
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинусная схожесть"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _find_boundaries(self, similarities: List[float]) -> List[int]:
        """
        Поиск границ чанков
        
        Граница ставится где:
        - Схожесть ниже порога И достигнут min_sentences
        - ИЛИ достигнут max_sentences
        """
        boundaries = []
        current_size = 1
        
        for i, sim in enumerate(similarities):
            # Проверка: достигнут максимум
            if current_size >= self.max_sentences:
                boundaries.append(i + 1)
                current_size = 0
            # Проверка: схожесть упала И достигнут минимум
            elif sim < self.similarity_threshold and current_size >= self.min_sentences:
                boundaries.append(i + 1)
                current_size = 0
            
            current_size += 1
        
        return boundaries
    
    def _build_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """
        Построение чанков из предложений
        ВАЖНО: Никогда не обрезаем предложения!
        """
        chunks = []
        start = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start:boundary]
            if chunk_sentences:
                # Объединяем предложения в чанк
                chunk = ' '.join(chunk_sentences)
                chunks.append(chunk)
            start = boundary
        
        # Последний чанк
        if start < len(sentences):
            chunk = ' '.join(sentences[start:])
            chunks.append(chunk)
        
        return chunks
    
    def process_file(self, file_path: str) -> List[str]:
        """Обработка файла"""
        logger.info(f"Обработка файла: {file_path}")
        
        # Чтение файла
        encodings = ['utf-8', 'cp1251', 'latin-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"✅ Прочитано (кодировка: {encoding})")
                break
            except:
                continue
        
        if not content:
            logger.error(f"❌ Не удалось прочитать файл")
            return []
        
        # Создаем чанки
        chunks = self.create_semantic_chunks(content)
        
        logger.info(f"✅ Создано {len(chunks)} чанков")
        return chunks
    
    def process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """Обработка директории"""
        results = {}
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory_path, filename)
                chunks = self.process_file(filepath)
                results[filename] = chunks
        
        return results


if __name__ == '__main__':
    # Тест
    processor = SemanticTextProcessor(
        similarity_threshold=0.45,
        min_sentences=2,
        max_sentences=10
    )
    
    test_text = """
    Евгений Онегин - роман в стихах. Это великое произведение.
    Пушкин написал его в 19 веке. Роман считается шедевром.
    
    Главный герой - Евгений Онегин. Он молодой дворянин.
    Онегин живет в Петербурге. Он ведет светскую жизнь.
    """
    
    chunks = processor.create_semantic_chunks(test_text)
    print(f"\nСоздано чанков: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Чанк {i} ({len(chunk)} символов) ---")
        print(chunk)
        # Проверка что заканчивается на точку
        if chunk and not chunk.strip().endswith(('.', '!', '?', '...')):
            print("⚠️ WARNING: Чанк не заканчивается на точку!")
