import os
import numpy as np
from typing import List, Tuple
import logging
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Импортируем необходимые классы из faiss
IndexFlatIP = faiss.IndexFlatIP
IndexIDMap = faiss.IndexIDMap

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """Класс для создания эмбеддингов с помощью Sentence Transformers и хранения в FAISS"""

    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """
        Инициализация процессора эмбеддингов

        Args:
            model_name: Название модели Sentence Transformers
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.id_map = {}  # chunk_id -> index_in_faiss
        self.chunk_paths = {}  # chunk_id -> путь к файлу чанка
        self.current_id = 0
        
    def create_embeddings(self, chunks: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Создание эмбеддингов для чанков

        Args:
            chunks: Список чанков текста

        Returns:
            Кортеж из массива эмбеддингов и списка ID
        """
        logger.info(f"Создание эмбеддингов для {len(chunks)} чанков")

        # Создание эмбеддингов
        embeddings = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        # Создание уникальных ID для каждого чанка
        ids = list(range(self.current_id, self.current_id + len(chunks)))
        self.current_id += len(chunks)

        logger.info(f"Создано {len(embeddings)} эмбеддингов")
        return embeddings, ids
    
    def initialize_faiss_index(self, dimension: int, metric: str = 'ip'):
        """
        Инициализация FAISS индекса
        
        Args:
            dimension: Размерность эмбеддингов
            metric: Метрика相似ности ('ip' для внутреннего произведения, 'l2' для Евклидова расстояния)
        """
        if metric == 'ip':
            self.index = IndexIDMap(IndexFlatIP(dimension))
        elif metric == 'l2':
            self.index = IndexIDMap(IndexFlatIP(dimension))
        else:
            raise ValueError("Метрика должна быть 'ip' или 'l2'")
        
        logger.info(f"Инициализирован FAISS индекс с размерностью {dimension}")
    
    def add_embeddings_to_index(self, embeddings: np.ndarray, ids: List[int], chunk_paths: List[str] = None):
       """
       Добавление эмбеддингов в FAISS индекс

       Args:
           embeddings: Массив эмбеддингов
           ids: Список ID для эмбеддингов
           chunk_paths: Список путей к файлам чанков (опционально)
       """
       if self.index is None:
           raise ValueError("FAISS индекс не инициализирован. Сначала вызовите initialize_faiss_index()")

       # Добавление в индекс с указанием ID
       self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

       # Обновление ID-карты и путей к чанкам
       for i, id_val in enumerate(ids):
           self.id_map[id_val] = i
           if chunk_paths and i < len(chunk_paths):
               self.chunk_paths[id_val] = chunk_paths[i]

       logger.info(f"Добавлено {len(embeddings)} эмбеддингов в индекс")
    
    def save_index(self, index_path: str, id_map_path: str = None):
        """
        Сохранение индекса и ID-карты

        Args:
            index_path: Путь для сохранения индекса
            id_map_path: Путь для сохранения ID-карты (если None, используется index_path + '_id_map.pkl')
        """
        if self.index is None:
            raise ValueError("Индекс не инициализирован")

        # Сохранение индекса
        faiss.write_index(self.index, index_path)

        # Сохранение ID-карты и путей к чанкам
        if id_map_path is None:
            id_map_path = index_path + '_id_map.pkl'

        with open(id_map_path, 'wb') as f:
            pickle.dump({'id_map': self.id_map, 'chunk_paths': self.chunk_paths}, f)

        logger.info(f"Индекс сохранен в {index_path}")
        logger.info(f"ID-карта сохранена в {id_map_path}")
    
    def load_index(self, index_path: str, id_map_path: str = None):
        """
        Загрузка индекса и ID-карты

        Args:
            index_path: Путь к сохраненному индексу
            id_map_path: Путь к сохраненной ID-карте (если None, используется index_path + '_id_map.pkl')
        """
        global faiss

        try:
            import faiss
        except ImportError:
            raise ImportError("Пожалуйста, установите faiss: pip install faiss-cpu")

        # Загрузка индекса
        self.index = faiss.read_index(index_path)

        # Загрузка ID-карты и путей к чанкам
        if id_map_path is None:
            id_map_path = index_path + '_id_map.pkl'

        with open(id_map_path, 'rb') as f:
            data = pickle.load(f)
            # Поддержка старого формата (только id_map)
            if isinstance(data, dict) and 'id_map' in data:
                self.id_map = data['id_map']
                self.chunk_paths = data.get('chunk_paths', {})
            else:
                self.id_map = data
                self.chunk_paths = {}

        logger.info(f"Индекс загружен из {index_path}")
        logger.info(f"ID-карта загружена из {id_map_path}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Поиск ближайших эмбеддингов к запросу

        Args:
            query: Поисковый запрос
            k: Количество результатов

        Returns:
            Список кортежей (ID, расстояние)
        """
        if self.index is None:
            raise ValueError("Индекс не инициализирован")

        # Создание эмбеддинга для запроса с нормализацией
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Поиск
        distances, indices = self.index.search(query_embedding, k)

        # Преобразование индексов в ID через id_map
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS возвращает -1 для паддинга
                # Находим ID по индексу (id_map: ID -> index)
                for chunk_id, faiss_index in self.id_map.items():
                    if faiss_index == idx:
                        results.append((chunk_id, float(distances[0][i])))
                        break

        return results

    def get_chunk_content(self, chunk_id: int) -> str:
        """
        Загрузка содержимого чанка по ID

        Args:
            chunk_id: ID чанка

        Returns:
            Содержимое чанка
        """
        if chunk_id not in self.chunk_paths:
            logger.warning(f"Путь к чанку {chunk_id} не найден")
            return None

        path = self.chunk_paths[chunk_id]
        if os.path.exists(path):
            # Пробуем разные кодировки
            encodings = ['utf-8', 'cp1251', 'latin-1']
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
            logger.warning(f"Не удалось прочитать файл {path} ни в одной из кодировок")
            return None
        else:
            logger.warning(f"Файл чанка не найден: {path}")
            return None

# Пример использования
if __name__ == "__main__":
    pass