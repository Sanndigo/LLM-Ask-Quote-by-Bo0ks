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
    
    def __init__(self, chunk_size: int = 256, overlap: int = 64):
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
        # Пробуем разные кодировки
        encodings = ['utf-8', 'cp1251', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                logger.info(f"Успешно прочитан файл: {file_path} (кодировка: {encoding})")
                return content
            except UnicodeDecodeError:
                continue
        
        # Если ни одна кодировка не подошла, выбрасываем ошибку
        logger.error(f"Не удалось прочитать файл {file_path} ни в одной из кодировок")
        raise ValueError(f"Не удалось прочитать файл {file_path}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста

        Args:
            text: Входной текст

        Returns:
            Обработанный текст
        """
        # Удаляем шапку библиотеки Royallibru в начале
        if 'royallibru' in text.lower()[:500]:
            # Ищем конец шапки - обычно после "Приятного чтения" или имени автора
            markers = ['Приятного чтения', 'notes Примечания', 'Роман в стихах', 'Повесть']
            for marker in markers:
                idx = text.find(marker)
                if idx != -1:
                    text = text[idx + len(marker):]
                    break
        
        # Сохраняем переносы строк для последующего разбиения на абзацы
        # Оставляем буквы (кириллицу и латиницу), цифры, пробелы и переносы строк
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s\n]', '', text)
        # Удаляем лишние пробелы (но не переносы строк)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения/абзацы

        Args:
            text: Входной текст

        Returns:
            Список предложений
        """
        # Сначала пробуем разбить по абзацам (для русского языка работает лучше)
        # Обрабатываем разные типы переносов строк
        text_normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        paragraphs = [p.strip() for p in text_normalized.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        if len(paragraphs) > 1:
            return paragraphs
        
        # Если абзацев нет, разбиваем по предложениям NLTK
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
        # Удаление стоп-слов (английский и русский)
        try:
            stop_words = set(stopwords.words('russian'))
            stop_words.update(stopwords.words('english'))
        except Exception:
            stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Лемматизация (только для английских слов)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
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

        # Разбиение на абзацы/предложения
        sentences = self.split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Считаем длину по символам (примерно 4 символа на слово)
            sentence_length = len(sentence) // 4  # Примерное количество слов
            
            # Если текущий чанк превышает размер, сохраняем его и начинаем новый
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Создаем перекрытие - берем последние N элементов
                if overlap > 0 and len(current_chunk) > 1:
                    overlap_count = min(overlap, len(current_chunk) - 1)
                    current_chunk = current_chunk[-overlap_count:]
                else:
                    current_chunk = []
                current_length = sum(len(s) // 4 for s in current_chunk)

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