import os
import re
from typing import List, Generator
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Класс для обработки текстовых файлов и разбиения на логические чанки"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Инициализация процессора текста
        
        Args:
            chunk_size: Размер чанка в токенах
            overlap: Перекрытие между чанками в токенах
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lemmatizer = WordNetLemmatizer()
        
        # Загрузка необходимых данных NLTK
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Загрузка необходимых данных NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def read_txt_file(self, file_path: str) -> str:
        """
        Чтение содержимого TXT файла
        
        Args:
            file_path: Путь к TXT файлу
            
        Returns:
            Содержимое файла как строка
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Успешно прочитан файл: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста
        
        Args:
            text: Входной текст
            
        Returns:
            Обработанный текст
        """
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        # Удаление специальных символов (оставляем только буквы, цифры и пробелы)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения
        
        Args:
            text: Входной текст
            
        Returns:
            Список предложений
        """
        sentences = sent_tokenize(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Токенизация текста
        
        Args:
            text: Входной текст
            
        Returns:
            Список токенов
        """
        tokens = word_tokenize(text.lower())
        # Удаление стоп-слов
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Лемматизация
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def create_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Создание логических чанков из текста
        
        Args:
            text: Входной текст
            chunk_size: Размер чанка (по умолчанию используется self.chunk_size)
            overlap: Перекрытие между чанками (по умолчанию используется self.overlap)
            
        Returns:
            Список чанков
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
            
        # Разбиение на предложения
        sentences = self.split_into_sentences(text)
        
        # Преобразование предложений в токены для подсчета длины
        sentence_tokens = [self.tokenize_text(sentence) for sentence in sentences]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, (sentence, tokens) in enumerate(zip(sentences, sentence_tokens)):
            sentence_length = len(tokens)
            
            # Если текущий чанк превышает размер, сохраняем его и начинаем новый
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Создаем перекрытие
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_length = sum(len(self.tokenize_text(s)) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Полный процесс обработки одного файла
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Список чанков текста
        """
        # Чтение файла
        raw_text = self.read_txt_file(file_path)
        
        # Предобработка
        processed_text = self.preprocess_text(raw_text)
        
        # Создание чанков
        chunks = self.create_chunks(processed_text)
        
        logger.info(f"Файл {file_path} обработан: {len(chunks)} чанков создано")
        return chunks
    
    def process_directory(self, directory_path: str) -> Generator[tuple, None, None]:
        """
        Обработка всех TXT файлов в директории
        
        Args:
            directory_path: Путь к директории
            
        Yields:
            Кортежи (путь_к_файлу, список_чанков)
        """
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    chunks = self.process_file(file_path)
                    yield (file_path, chunks)
                except Exception as e:
                    logger.error(f"Ошибка при обработке файла {file_path}: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Создание экземпляра процессора
    processor = TextProcessor(chunk_size=512, overlap=50)
    
    # Пример обработки одного файла
    # chunks = processor.process_file("example.txt")
    # print(f"Создано {len(chunks)} чанков")
    
    # Пример обработки директории
    # for file_path, chunks in processor.process_directory("./texts"):
    #     print(f"Файл: {file_path}")
    #     print(f"Чанков: {len(chunks)}")