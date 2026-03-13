#!/usr/bin/env python3
"""
RAG-система для работы с книгами
Поиск фрагментов и ответы на вопросы с цитатами

Поддерживает:
- Локальные LLM (Qwen, Phi-3, Gemma)
- GigaChat API
- YandexGPT API
"""
import os
import sys
import json
import requests
from typing import List, Dict, Optional
from embedding_processor import EmbeddingProcessor
from text_processor import TextProcessor
import logging
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Настройка логирования
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GigaChatAPI:
    """Клиент для GigaChat API"""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id or os.getenv('GIGACHAT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('GIGACHAT_CLIENT_SECRET')
        self.base_url = 'https://gigachat.devices.sberbank.ru/api/v2'
        self.token = None
        self.token_expires = 0
        
    def _get_token(self) -> str:
        """Получение токена доступа"""
        import time
        if self.token and time.time() < self.token_expires:
            return self.token
            
        auth_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'RqUID': '00000000-0000-0000-0000-000000000000'
        }
        data = {
            'scope': 'GIGACHAT_API_PERS',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        try:
            response = requests.post(auth_url, headers=headers, data=data, verify=False)
            if response.status_code == 200:
                token_data = response.json()
                self.token = token_data['access_token']
                self.token_expires = time.time() + token_data['expires_in'] - 60
                return self.token
            else:
                logger.error(f"GigaChat auth error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"GigaChat auth error: {e}")
            return None
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Генерация ответа через GigaChat"""
        if not self.client_id or not self.client_secret:
            return None
            
        token = self._get_token()
        if not token:
            return None
        
        url = f'{self.base_url}/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        payload = {
            'model': 'GigaChat',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': 0.5
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, verify=False)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"GigaChat API error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"GigaChat API error: {e}")
            return None


class BookRAG:
    """RAG-система для работы с книгами"""

    def __init__(
        self,
        index_path: str = 'embeddings/faiss_index.bin',
        id_map_path: str = 'embeddings/faiss_index.bin_id_map.pkl',
        model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        llm_model_name: str = 'GIGACHAT',  # 'GIGACHAT' или 'Qwen/Qwen2.5-0.5B-Instruct'
        gigachat_client_id: str = None,
        gigachat_client_secret: str = None
    ):
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.embedding_processor = None
        self.loaded = False
        self.llm_model = None
        self.tokenizer = None
        self.last_results = []
        
        # GigaChat API
        self.gigachat = None
        if llm_model_name == 'GIGACHAT':
            self.gigachat = GigaChatAPI(gigachat_client_id, gigachat_client_secret)
            logger.info("✅ GigaChat API инициализирован")
        else:
            # Загрузка локальной модели
            self.embedder = SentenceTransformer(model_name)
            self._load_llm_model()

    def _load_llm_model(self):
        """Загрузка LLM модели"""
        try:
            logger.info(f"Загрузка LLM {self.llm_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
            
            # Загрузка модели
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16 if gpu_memory >= 4 else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info(f"✅ LLM загружена на GPU: {self.llm_model_name}")
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info(f"✅ LLM загружена на CPU: {self.llm_model_name}")
            
            self.llm_model.eval()
            
        except Exception as e:
            logger.error(f"❌ Не удалось загрузить {self.llm_model_name}: {e}")
            logger.error("Установите модель которая помещается в вашу память!")
            raise

    def load_index(self):
        """Загрузка индекса"""
        self.embedding_processor = EmbeddingProcessor(self.model_name)
        self.embedding_processor.load_index(self.index_path, self.id_map_path)
        self.loaded = True
        logger.info("Индекс загружен")

    def _get_book_content(self, book_name: str) -> str:
        """
        Загрузка полного текста книги по имени
        
        Args:
            book_name: Имя книги (например 'EvgeniyOnegin')
        
        Returns:
            Полный текст книги или пустая строка
        """
        # Сопоставление имени файла с книгой
        book_files = {
            'EvgeniyOnegin': 'Евгений Онегин',
            'Shinell': 'Шинель',
            'VlastelinKolec': 'Властелин Колец',
            'sample': 'NLP Статья'
        }
        
        # Ищем файлы в processed
        processed_dir = 'processed'
        if not os.path.exists(processed_dir):
            return ""
        
        # Находим все чанки для этой книги
        book_chunks = []
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                if file.startswith(book_name) and file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content and not content.startswith('[Контент недоступен'):
                                book_chunks.append((file, content))
                    except Exception as e:
                        logger.warning(f"Не удалось прочитать {filepath}: {e}")
        
        # Сортируем по номеру чанка
        def get_chunk_num(filename):
            try:
                return int(filename.replace('.txt', '').split('_chunk_')[-1])
            except:
                return 999
        
        book_chunks.sort(key=lambda x: get_chunk_num(x[0]))
        
        # Объединяем тексты
        full_text = "\n\n".join([chunk[1] for chunk in book_chunks])
        return full_text[:15000]  # Ограничиваем 15K токенов

    def _detect_book_from_fragments(self, fragments: List[Dict]) -> Optional[str]:
        """
        Определение основной книги по фрагментам
        
        Returns:
            Имя книги или None
        """
        if not fragments:
            return None
        
        # Считаем какая книга чаще встречается
        book_counts = {}
        for frag in fragments:
            book = frag.get('source', 'Unknown')
            book_counts[book] = book_counts.get(book, 0) + 1
        
        # Возвращаем самую частую
        if book_counts:
            main_book = max(book_counts, key=book_counts.get)
            if book_counts[main_book] >= len(fragments) * 0.6:  # 60%+ фрагментов
                return main_book
        return None

    def search_fragments(self, query: str, k: int = 5) -> List[Dict]:
        """
        Поиск фрагментов по запросу
        
        Returns:
            Список фрагментов с источниками
        """
        if not self.loaded:
            self.load_index()

        results = self.embedding_processor.search(query, k)
        
        fragments = []
        for chunk_id, distance in results:
            content = self.embedding_processor.get_chunk_content(chunk_id)
            
            # Извлекаем информацию об источнике
            source_info = self._extract_source_info(chunk_id, content)
            
            if content:
                fragments.append({
                    'chunk_id': chunk_id,
                    'similarity': 1.0 - distance,
                    'content': content[:400] + ('...' if len(content) > 400 else ''),
                    'source': source_info['book'],
                    'position': source_info['position'],
                    'chapter': source_info['chapter'],
                    'full_content': content
                })

        # Сортировка по схожести
        fragments.sort(key=lambda x: x['similarity'], reverse=True)
        self.last_results = fragments
        return fragments

    def _extract_source_info(self, chunk_id: int, content: str) -> Dict:
        """
        Извлечение информации об источнике из контента
        
        Returns:
            Словарь с книгой, позицией и главой
        """
        book_name = "Неизвестно"
        position = ""
        chapter = ""
        
        if chunk_id in self.embedding_processor.chunk_paths:
            path = self.embedding_processor.chunk_paths[chunk_id]
            filename = os.path.basename(path)
            parts = filename.replace('.txt', '').split('_chunk_')
            if len(parts) == 2:
                book_name = parts[0]
                chunk_num = int(parts[1])
                position = f"фрагмент #{chunk_num + 1}"
        
        # Пытаемся найти название книги/главы в начале контента
        if content:
            lines = content.split('\n')[:5]  # Первые 5 строк
            
            # Определяем название книги по имени файла
            if 'EvgeniyOnegin' in book_name or 'evgeniy' in book_name.lower():
                book_name = "Евгений Онегин"
            elif 'Shinell' in book_name or 'shinell' in book_name.lower():
                book_name = "Шинель"
            elif 'sample' in book_name.lower():
                book_name = "NLP Статья (англ.)"
            
            # Ищем номер главы/части в первых строках
            import re
            for line in lines:
                line_clean = line.strip()
                if len(line_clean) < 10 or len(line_clean) > 80:
                    continue
                    
                # Пропускаем URL и технические надписи
                if 'http' in line_clean.lower() or 'royal' in line_clean.lower():
                    continue
                
                # Паттерны для глав: "Глава 1", "Часть I", "Песнь V" и т.д.
                chapter_patterns = [
                    (r'[Гг]лава\s*([IVX0-9]+)', 'Глава'),
                    (r'[Чч]асть\s*([IVX0-9]+)', 'Часть'),
                    (r'[Пп]еснь\s*([IVX0-9]+)', 'Песнь'),
                    (r'[Сс]трофа\s*([IVX0-9]+)', 'Строфа'),
                    (r'^([IVX]{1,3})\s*$', '№'),  # Просто римская цифра I-XXX
                ]
                
                for pattern, prefix in chapter_patterns:
                    match = re.search(pattern, line_clean)
                    if match:
                        chapter_num = match.group(1)
                        chapter = f"{prefix} {chapter_num}"
                        break
                
                # Если уже нашли главу, прекращаем
                if chapter:
                    break
        
        # Формируем красивую позицию
        if chapter and position:
            position = f"{chapter}, {position}"
        elif chapter:
            position = chapter
        
        return {
            'book': book_name,
            'position': position if position else "фрагмент неизвестен",
            'chapter': chapter
        }

    def answer_question(self, question: str, k: int = 5) -> Dict:
        """
        Ответ на вопрос с цитатами
        
        Returns:
            Словарь с ответом и цитатами
        """
        # Поиск релевантных фрагментов
        fragments = self.search_fragments(question, k)
        
        if not fragments or fragments[0]['similarity'] < 0.25:
            return {
                'answer': "К сожалению, в загруженных текстах нет информации, которая могла бы ответить на ваш вопрос.",
                'quotes': [],
                'found': False
            }
        
        # Определяем основную книгу
        main_book = self._detect_book_from_fragments(fragments)
        book_context = ""
        
        if main_book:
            # Загружаем полный текст книги
            book_context = self._get_book_content(main_book)
            logger.info(f"📚 Загружен текст книги: {main_book} ({len(book_context)} символов)")
        
        # Формируем контекст из фрагментов
        context_parts = []
        quotes = []
        
        for i, frag in enumerate(fragments[:5], 1):
            source_detail = f"{frag['source']}"
            if frag.get('position') and frag['position'] != "фрагмент неизвестен":
                source_detail += f" ({frag['position']})"
            
            context_parts.append(f"[{source_detail}]: {frag['full_content'][:300]}")
            quotes.append({
                'text': frag['content'],
                'source': source_detail
            })
        
        fragments_context = "\n\n".join(context_parts)

        # Если GigaChat - используем API
        if self.gigachat:
            prompt = f"""Ты — литературный эксперт. Отвечай ТОЛЬКО на основе контекста.

КОНТЕКСТ ИЗ КНИГ:
{book_context[:8000] if book_context else fragments_context}

ВОПРОС: {question}

ОТВЕТ (2-4 предложения, по-русски):"""
            
            answer = self.gigachat.generate(prompt, max_tokens=500)
            if answer:
                return {
                    'answer': answer,
                    'quotes': quotes,
                    'found': True
                }
            else:
                return {
                    'answer': "Ошибка при обращении к GigaChat API. Проверьте ключи доступа.",
                    'quotes': quotes,
                    'found': False
                }
        
        # Если LLM не загружена - возвращаем только цитаты
        if self.llm_model is None or self.tokenizer is None:
            return {
                'answer': f"Найденные фрагменты по вашему запросу:\n\n{fragments_context}",
                'quotes': quotes,
                'found': True
            }
        
        # Генерация ответа через LLM
        system_prompt = """Ты — литературный эксперт. Отвечай ТОЛЬКО на основе предоставленного контекста.

КРИТИЧЕСКИ ВАЖНО:
1. ЕСЛИ в контексте НЕТ ответа — скажи "В загруженных книгах нет информации об этом"
2. НЕ выдумывай факты, имена, события
3. ИСПОЛЬЗУЙ только информацию из КОНТЕКСТА выше
4. ЦИТИРУЙ книгу когда отвечаешь
5. Кратко: 2-4 предложения

Пример правильного ответа:
"Согласно тексту из книги [название], [факт из контекста]."""

        # Формируем промпт с полным текстом книги если есть
        if book_context:
            user_prompt = f"""КНИГА: {main_book}
ПОЛНЫЙ ТЕКСТ (для контекста):
{book_context[:10000]}

НАЙДЕННЫЕ ФРАГМЕНТЫ (наиболее релевантные):
{fragments_context}

ВОПРОС: {question}

ОТВЕТ (используй полный текст книги + фрагменты):"""
        else:
            user_prompt = f"""КОНТЕКСТ ИЗ КНИГ (используй ТОЛЬКО это):
{fragments_context}

ВОПРОС: {question}

ОТВЕТ (только по контексту выше):"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Очищаем ответ — берем текст после последнего маркера
            for marker in ["ОТВЕТ (только по контексту выше):", "ОТВЕТ:", "Ответ:"]:
                if marker in response:
                    response = response.split(marker)[-1].strip()
                    break

            # Удаляем префиксы и мусор
            lines = response.split('\n')
            clean_lines = []
            skip_words = ['assistant', 'user', 'system', 'КОНТЕКСТ', 'ВОПРОС', 'Правила', 'Пример']
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if any(word in stripped for word in skip_words):
                    continue
                if stripped.startswith(('1.', '2.', '3.', '4.', '5.')) and 'КРИТИЧЕСКИ' in response[:200]:
                    continue
                clean_lines.append(line)
            
            response = '\n'.join(clean_lines).strip()
            
            # Если ответ пустой или бред — показываем цитаты
            if not response or len(response) < 20:
                if quotes and len(quotes) > 0:
                    response = "📖 Найденные фрагменты:\n\n"
                    for i, q in enumerate(quotes[:3], 1):
                        response += f"{i}. {q['source']}:\n   «{q['text'][:200]}...»\n\n"
                else:
                    response = "❌ К сожалению, не удалось найти информацию."
            
            # Ограничиваем длину
            if len(response) > 1000:
                response = response[:1000] + "..."

            return {
                'answer': response,
                'quotes': quotes,
                'found': True
            }

        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return {
                'answer': f"Ошибка при генерации ответа. Найденные фрагменты:\n\n{context_text}",
                'quotes': quotes,
                'found': True
            }


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_fragments(fragments: List[Dict]):
    if not fragments:
        print("\n❌ Ничего не найдено")
        return
    
    print(f"\n📚 Найдено фрагментов: {len(fragments)}")
    print("-" * 70)
    
    for i, frag in enumerate(fragments, 1):
        print(f"\n{i}. 📖 {frag['source']}")
        if frag.get('position') and frag['position'] != "фрагмент неизвестен":
            print(f"   📍 {frag['position']}")
        print(f"   📊 Схожесть: {frag['similarity']:.1%}")
        print(f"   📝 {frag['content']}\n")


def print_answer(result: Dict):
    print_header("ОТВЕТ")
    print(result['answer'])
    
    if result.get('quotes'):
        print("\n" + "-" * 70)
        print("📑 ЦИТАТЫ (источники):")
        print("-" * 70)
        for i, quote in enumerate(result['quotes'], 1):
            print(f"\n{i}. 📖 {quote['source']}")
            print(f"   «{quote['text']}»")


def interactive_mode():
    print_header("📚 RAG-СИСТЕМА ДЛЯ РАБОТЫ С КНИГАМИ")
    print("Режимы работы:")
    print("  1 — Поиск фрагментов текста")
    print("  2 — Ответ на вопрос по книгам")
    print("  quit — Выход")
    print("  help — Справка")

    rag = BookRAG()
    rag.load_index()
    print("\n✅ Система готова к работе")

    while True:
        try:
            choice = input("\nВыберите режим (1/2): ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q', 'выход']:
                print("\n👋 До свидания!")
                break
            
            if choice.lower() in ['help', 'h', 'справка']:
                print_header("СПРАВКА")
                print("1. Поиск фрагментов — находит отрывки текста по описанию")
                print("   Пример: «где Татьяна пишет письмо»")
                print("\n2. Ответ на вопрос — отвечает на вопрос с цитатами")
                print("   Пример: «что писала Наташа Евгению Онегину»")
                continue
            
            if choice == '1':
                query = input("\n🔍 Что найти: ").strip()
                if not query:
                    continue
                
                print("\n⏳ Поиск...")
                fragments = rag.search_fragments(query, k=5)
                print_fragments(fragments)
                
            elif choice == '2':
                question = input("\n❓ Ваш вопрос: ").strip()
                if not question:
                    continue
                
                print("\n⏳ Поиск и генерация ответа...")
                result = rag.answer_question(question, k=5)
                print_answer(result)
                
            else:
                print("⚠️ Выберите 1 или 2")

        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except EOFError:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        interactive_mode()
