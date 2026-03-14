#!/usr/bin/env python3
"""
Основной скрипт для обработки TXT файлов
Использует семантический чанкинг на основе эмбеддингов
"""
import os
import sys
import argparse
from pathlib import Path
from semantic_processor import SemanticTextProcessor
from embedding_processor import EmbeddingProcessor
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Создание необходимых директорий"""
    directories = ['data', 'processed', 'embeddings']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Созданы необходимые директории")

def process_txt_files(input_dir: str, output_dir: str, semantic_threshold: float = 0.45):
    """
    Обработка TXT файлов с использованием семантического чанкинга

    Args:
        input_dir: Входная директория с TXT файлами
        output_dir: Выходная директория для сохранения чанков
        semantic_threshold: Порог схожести для семантического чанкинга
    """
    logger.info(f"Начинается семантическая обработка файлов из {input_dir}")

    # Создание процессора
    semantic_processor = SemanticTextProcessor(
        similarity_threshold=semantic_threshold,
        min_chunk_size=3,
        max_chunk_size=15
    )

    # Создание директории для сохранения чанков
    os.makedirs(output_dir, exist_ok=True)

    # Обработка всех TXT файлов
    total_chunks = 0
    processed_files = 0

    for filename, chunks in semantic_processor.process_directory(input_dir).items():
        # Сохранение чанков в отдельные файлы
        name_without_ext = os.path.splitext(filename)[0]

        # Создание поддиректории для каждого файла
        file_output_dir = os.path.join(output_dir, name_without_ext)
        os.makedirs(file_output_dir, exist_ok=True)

        # Сохранение чанков
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{name_without_ext}_chunk_{i}.txt"
            chunk_path = os.path.join(file_output_dir, chunk_filename)

            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)

        total_chunks += len(chunks)
        processed_files += 1
        logger.info(f"Обработан файл {filename}: {len(chunks)} семантических чанков")

    logger.info(f"Обработка завершена: {processed_files} файлов, {total_chunks} чанков")

def create_embeddings(input_dir: str, output_dir: str, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
    """
    Создание эмбеддингов из чанков

    Args:
        input_dir: Входная директория с чанками
        output_dir: Выходная директория для сохранения эмбеддингов
        model_name: Название модели для создания эмбеддингов
    """
    logger.info(f"Создание эмбеддингов из {input_dir}")

    # Создание процессора эмбеддингов
    embedding_processor = EmbeddingProcessor(model_name=model_name)

    # Сбор всех чанков
    all_chunks = []
    chunk_paths = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk_content = f.read().strip()
                    if chunk_content:
                        all_chunks.append(chunk_content)
                        chunk_paths.append(file_path)

    logger.info(f"Собрано {len(all_chunks)} чанков для создания эмбеддингов")

    if not all_chunks:
        logger.warning("Нет чанков для создания эмбеддингов")
        return

    # Создание эмбеддингов
    embeddings, ids = embedding_processor.create_embeddings(all_chunks)

    # Инициализация FAISS индекса
    embedding_processor.initialize_faiss_index(embeddings.shape[1])

    # Добавление эмбеддингов в индекс с путями к чанкам
    embedding_processor.add_embeddings_to_index(embeddings, ids, chunk_paths)

    # Сохранение индекса
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'faiss_index.bin')
    embedding_processor.save_index(index_path)

    logger.info(f"Эмбеддинги успешно созданы и сохранены в {index_path}")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Обработка TXT файлов с семантическим чанкингом')
    parser.add_argument('--input-dir', '-i', default='data', help='Входная директория с TXT файлами')
    parser.add_argument('--output-dir', '-o', default='processed', help='Выходная директория')
    parser.add_argument('--threshold', '-t', type=float, default=0.45, help='Порог схожести для семантического чанкинга')
    parser.add_argument('--model', '-m', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='Модель для создания эмбеддингов')
    parser.add_argument('--step', '-s', choices=['process', 'embed', 'all'], default='all', help='Этап выполнения')

    args = parser.parse_args()

    # Создание необходимых директорий
    setup_directories()

    if args.step in ['process', 'all']:
        # Обработка TXT файлов с семантическим чанкингом
        process_txt_files(args.input_dir, args.output_dir, args.threshold)

    if args.step in ['embed', 'all']:
        # Создание эмбеддингов
        create_embeddings(args.output_dir, 'embeddings', args.model)

    logger.info("Процесс завершен успешно")

if __name__ == "__main__":
    main()
