#!/usr/bin/env python3
"""
Семантический чанкинг текста на основе эмбеддингов
Разбивает текст по смысловым границам, а не по количеству токенов
"""
import os
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticTextProcessor:
    """
    Процессор для семантического чанкинга
    
    Принцип работы:
    1. Разбиваем текст на предложения
    2. Считаем эмбеддинги для каждого предложения
    3. Находим границы где схожесть между предложениями резко падает
    4. Объединяем предложения в чанки по смыслу
    """
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        similarity_threshold: float = 0.45,  # Порог схожести (0-1)
        min_chunk_size: int = 3,  # Мин кол-во предложений в чанке
        max_chunk_size: int = 15,  # Макс кол-во предложений в чанке
    ):
        """
        Args:
            model_name: Модель для эмбеддингов
            similarity_threshold: Порог схожести (ниже - новая тема)
            min_chunk_size: Минимум предложений в чанке
            max_chunk_size: Максимум предложений в чанке
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        logger.info(f"Загрузка модели: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        logger.info("✅ Модель загружена")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения
        
        Учитывает:
        - Точки, восклицательные, вопросительные знаки
        - Переносы строк
        - Особые случаи (г-н, и т.д.)
        """
        # Очищаем текст
        text = re.sub(r'\s+', ' ', text)
        
        # Разбиваем по предложениям
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[А-ЯA-Z])',
            text
        )
        
        # Фильтруем пустые и слишком короткие
        sentences = [
            s.strip() for s in sentences
            if s.strip() and len(s.strip()) > 10
        ]
        
        return sentences
    
    def compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        Вычисление схожести между соседними предложениями
        
        Returns:
            Список схожестей между i и i+1 предложением
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        return similarities
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинусная схожесть двух векторов"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """
        Создание семантических чанков
        
        Алгоритм:
        1. Разбиваем на предложения
        2. Считаем эмбеддинги
        3. Находим границы по схожести
        4. Формируем чанки
        """
        # 1. Разбиваем на предложения
        sentences = self.split_into_sentences(text)
        
        if len(sentences) < 2:
            return [text] if text else []
        
        logger.debug(f"Разбито на {len(sentences)} предложений")
        
        # 2. Считаем эмбеддинги
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
        
        # 3. Вычисляем схожести
        similarities = self.compute_similarities(embeddings)
        
        # 4. Находим границы чанков
        chunk_boundaries = self._find_boundaries(similarities)
        
        # 5. Формируем чанки
        chunks = self._build_chunks(sentences, chunk_boundaries)
        
        logger.info(f"Создано {len(chunks)} семантических чанков")
        return chunks
    
    def _find_boundaries(self, similarities: List[float]) -> List[int]:
        """
        Поиск границ чанков на основе схожести
        
        Граница ставится где:
        - Схожесть ниже порога
        - ИЛИ достигнут max_chunk_size
        """
        boundaries = []
        current_chunk_size = 1
        
        for i, sim in enumerate(similarities):
            # Проверка по схожести
            if sim < self.similarity_threshold:
                if current_chunk_size >= self.min_chunk_size:
                    boundaries.append(i + 1)
                    current_chunk_size = 0
            
            # Проверка по размеру
            current_chunk_size += 1
            if current_chunk_size >= self.max_chunk_size:
                boundaries.append(i + 1)
                current_chunk_size = 0
        
        return boundaries
    
    def _build_chunks(
        self,
        sentences: List[str],
        boundaries: List[int]
    ) -> List[str]:
        """
        Построение чанков из предложений по границам
        """
        chunks = []
        start = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start:boundary]
            if chunk_sentences:
                chunk = ' '.join(chunk_sentences)
                chunks.append(chunk)
            start = boundary
        
        # Последний чанк
        if start < len(sentences):
            chunk = ' '.join(sentences[start:])
            chunks.append(chunk)
        
        return chunks
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Обработка файла и создание семантических чанков
        """
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
            except Exception as e:
                continue
        
        if not content:
            logger.error(f"❌ Не удалось прочитать файл")
            return []
        
        # Создание чанков
        chunks = self.create_semantic_chunks(content)
        
        logger.info(f"✅ Создано {len(chunks)} чанков")
        return chunks
    
    def process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Обработка всех TXT файлов в директории
        
        Returns:
            Словарь {filename: [chunks]}
        """
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
        min_chunk_size=3,
        max_chunk_size=15
    )
    
    # Пример
    test_text = """
    Евгений Онегин - роман в стихах Александра Сергеевича Пушкина.
    Это произведение считается одним из величайших произведений русской литературы.
    
    Главный герой - Евгений Онегин, молодой дворянин.
    Он получает типичное для дворянства образование.
    Ведет светскую жизнь в Петербурге.
    """
    
    chunks = processor.create_semantic_chunks(test_text)
    print(f"\nСоздано чанков: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Чанк {i} ---")
        print(chunk[:200])
