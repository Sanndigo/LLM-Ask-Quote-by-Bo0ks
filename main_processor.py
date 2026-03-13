#!/usr/bin/env python3
"""
Основной скрипт для обработки TXT файлов, создания чанков и эмбеддингов
"""
import os
import sys
import argparse
from pathlib import Path
from text_processor import TextProcessor
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

def process_txt_files(input_dir: str, output_dir: str, chunk_size: int = 512, overlap: int = 50):
    """
    Обработка TXT файлов из директории
    
    Args:
        input_dir: Входная директория с TXT файлами
        output_dir: Выходная директория для сохранения чанков
        chunk_size: Размер чанка
        overlap: Перекрытие между чанками
    """
    logger.info(f"Начинается обработка файлов из {input_dir}")
    
    # Создание процессора текста
    text_processor = TextProcessor(chunk_size=chunk_size, overlap=overlap)
    
    # Создание директории для сохранения чанков
    os.makedirs(output_dir, exist_ok=True)
    
    # Обработка всех TXT файлов
    total_chunks = 0
    processed_files = 0
    
    for file_path, chunks in text_processor.process_directory(input_dir):
        # Сохранение чанков в отдельные файлы
        filename = os.path.basename(file_path)
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
        logger.info(f"Обработан файл {filename}: {len(chunks)} чанков")
    
    logger.info(f"Обработка завершена: {processed_files} файлов, {total_chunks} чанков")

def create_embeddings(input_dir: str, output_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
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

    # Сбор всех чанков с путями к файлам
    all_chunks = []
    chunk_paths = []

    # Функция для чтения файла с автоопределением кодировки
    def read_file_auto_encoding(file_path):
        encodings = ['utf-8', 'cp1251', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue
        logger.warning(f"Не удалось прочитать файл {file_path}")
        return None

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                chunk_content = read_file_auto_encoding(file_path)
                if chunk_content:  # Только непустые чанки
                    all_chunks.append(chunk_content)
                    chunk_paths.append(file_path)

    logger.info(f"Собрано {len(all_chunks)} чанков для создания эмбеддингов")

    # Проверка наличия чанков
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
    parser = argparse.ArgumentParser(description='Обработка TXT файлов для поиска с помощью Sentence Transformers и FAISS')
    parser.add_argument('--input-dir', '-i', default='data', help='Входная директория с TXT файлами')
    parser.add_argument('--output-dir', '-o', default='processed', help='Выходная директория')
    parser.add_argument('--chunk-size', '-c', type=int, default=1024, help='Размер чанка (в токенах)')
    parser.add_argument('--overlap', '-v', type=int, default=256, help='Перекрытие между чанками (в токенах)')
    parser.add_argument('--model', '-m', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='Модель для создания эмбеддингов')
    parser.add_argument('--step', '-s', choices=['process', 'embed', 'all'], default='all', help='Этап выполнения')
    
    args = parser.parse_args()
    
    # Создание необходимых директорий
    setup_directories()
    
    if args.step in ['process', 'all']:
        # Обработка TXT файлов
        process_txt_files(args.input_dir, args.output_dir, args.chunk_size, args.overlap)
    
    if args.step in ['embed', 'all']:
        # Создание эмбеддингов
        create_embeddings(args.output_dir, 'embeddings', args.model)
    
    logger.info("Процесс завершен успешно")

if __name__ == "__main__":
    main()