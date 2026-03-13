#!/usr/bin/env python3
"""
RAG-система для работы с книгами
Поиск фрагментов и ответы на вопросы с цитатами

Поддерживает:
- Локальные LLM (Qwen, Phi-3, Gemma)
- YandexGPT API (через OpenAI SDK)
"""
import os
import sys
import json
from typing import List, Dict, Optional
from embedding_processor import EmbeddingProcessor
from text_processor import TextProcessor
import logging
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YandexGPTAPI:
    """Клиент для YandexGPT API через OpenAI SDK"""
    
    def __init__(self, api_key: str = None, folder_id: str = None):
        self.api_key = api_key or os.getenv('YANDEXGPT_API_KEY')
        self.folder_id = folder_id or os.getenv('YANDEXGPT_FOLDER_ID')
        self.model = f"gpt://{self.folder_id}/yandexgpt/latest"
        self.client = None
        
        # Отладка
        if self.api_key:
            logger.info(f"✅ YandexGPT API Key: {self.api_key[:8]}...")
        else:
            logger.error("❌ YandexGPT API Key не найден!")
        
        if self.folder_id:
            logger.info(f"✅ YandexGPT Folder ID: {self.folder_id}")
        else:
            logger.error("❌ YandexGPT Folder ID не найден!")
        
        # Инициализация OpenAI клиента
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://ai.api.cloud.yandex.net/v1",
                project=self.folder_id
            )
            logger.info("✅ OpenAI клиент инициализирован для YandexGPT")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации OpenAI клиента: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.7) -> str:
        """Генерация ответа через YandexGPT"""
        if not self.client or not self.api_key or not self.folder_id:
            return None
        
        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=temperature,
                input=prompt,
                max_output_tokens=max_tokens
            )
            return response.output_text
        except Exception as e:
            logger.error(f"YandexGPT API error: {e}")
            return None


class BookRAG:
    """RAG-система для работы с книгами"""

    def __init__(
        self,
        index_path: str = 'embeddings/faiss_index.bin',
        id_map_path: str = 'embeddings/faiss_index.bin_id_map.pkl',
        model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        llm_model_name: str = 'YANDEXGPT',  # 'YANDEXGPT' или 'Qwen/Qwen2.5-0.5B-Instruct'
        yandexgpt_api_key: str = None,
        yandexgpt_folder_id: str = None
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
        
        # YandexGPT API
        self.yandexgpt = None
        if llm_model_name == 'YANDEXGPT':
            self.yandexgpt = YandexGPTAPI(yandexgpt_api_key, yandexgpt_folder_id)
            logger.info("✅ YandexGPT API инициализирован")
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
            Словарь с книгой, позицией, главой и частью
        """
        book_name = "Неизвестно"
        position = ""
        chapter = ""
        part = ""
        
        if chunk_id in self.embedding_processor.chunk_paths:
            path = self.embedding_processor.chunk_paths[chunk_id]
            filename = os.path.basename(path)
            parts = filename.replace('.txt', '').split('_chunk_')
            if len(parts) == 2:
                book_name = parts[0]
                chunk_num = int(parts[1])
                position = f"фрагмент #{chunk_num + 1}"
        
        # Пытаемся найти название книги/главы в контенте
        if content:
            # Берем первые 50 строк для поиска глав
            lines = content.split('\n')[:50]
            
            # Определяем название книги по имени файла
            if 'EvgeniyOnegin' in book_name or 'evgeniy' in book_name.lower():
                book_name = "Евгений Онегин"
            elif 'Shinell' in book_name or 'shinell' in book_name.lower():
                book_name = "Шинель"
            elif 'VlastelinKolec' in book_name or 'vlastelin' in book_name.lower():
                book_name = "Властелин Колец"
            elif 'sample' in book_name.lower():
                book_name = "NLP Статья"
            
            # Ищем главы, части, строфы в тексте
            import re
            for line in lines:
                line_clean = line.strip()
                if len(line_clean) < 2 or len(line_clean) > 100:
                    continue
                    
                # Пропускаем URL и технические надписи
                skip_patterns = ['http', 'royal', 'скачали', 'библиотек', 'автора', 'книги', 'форматах']
                if any(p in line_clean.lower() for p in skip_patterns):
                    continue
                
                # Паттерны для глав, частей, строф
                chapter_patterns = [
                    # Строфа XXV, Строфа V, Строфа I
                    (r'[Сс]трофа\s+([IVX]+)', 'Строфа'),
                    # Глава I, Глава 1
                    (r'[Гг]лава\s+([IVX0-9]+)', 'Глава'),
                    # Часть I, Часть 1
                    (r'[Чч]асть\s+([IVX0-9]+)', 'Часть'),
                    # Песнь V, Песнь 5
                    (r'[Пп]еснь\s+([IVX0-9]+)', 'Песнь'),
                    # Отдельная римская цифра в начале строки (I, II, III, IV, V и т.д.)
                    (r'^([IVX]{1,3})\s*$', 'Строфа'),
                ]
                
                for pattern, prefix in chapter_patterns:
                    match = re.search(pattern, line_clean)
                    if match:
                        num = match.group(1)
                        chapter = f"{prefix} {num}"
                        break
                
                # Если уже нашли главу, прекращаем
                if chapter:
                    break
        
        # Формируем красивую позицию
        if chapter:
            position = chapter
        
        return {
            'book': book_name,
            'position': position if position else "фрагмент неизвестен",
            'chapter': chapter,
            'part': part
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

        # Если YandexGPT - используем API
        if self.yandexgpt:
            prompt = f"""Ты — литературный эксперт. Отвечай на вопросы на основе контекста.

Правила:
1. Используй контекст из книг для ответа
2. УКАЗЫВАЙ главы, строфы, части если они есть в контексте (например: "в Строфе XXV говорится...")
3. Давай развернутые ответы (5-8 предложений)
4. Будь естественным и дружелюбным
5. Отвечай на русском языке

КОНТЕКСТ ИЗ КНИГ:
{book_context[:8000] if book_context else fragments_context}

ВОПРОС: {question}

ОТВЕТ (развернутый, 5-8 предложений, упоминай главы/строфы):"""

            answer = self.yandexgpt.generate(prompt, max_tokens=1500, temperature=0.7)
            if answer:
                return {
                    'answer': answer,
                    'quotes': quotes,
                    'found': True
                }
            else:
                return {
                    'answer': "Ошибка при обращении к YandexGPT API. Проверьте ключи доступа.",
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
5. Подробно: 3-4 предложения

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
                'answer': f"Ошибка при генерации ответа. Найденные фрагменты:\n\n{fragments_context}",
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
        
        # Показываем номер фрагмента + главу если найдена
        frag_num = f"Фрагмент #{frag['chunk_id'] + 1}"
        if frag.get('chapter') and frag['chapter'] != "фрагмент неизвестен":
            print(f"   📍 {frag_num} | {frag['chapter']}")
        elif frag.get('position') and frag['position'] != "фрагмент неизвестен":
            print(f"   📍 {frag_num} | {frag['position']}")
        else:
            print(f"   📍 {frag_num}")
        
        print(f"   📊 Схожесть: {frag['similarity']:.1%}")
        print(f"   📝 {frag['content']}\n")


def print_answer(result: Dict):
    print("\n" + "=" * 70)
    print(" 💬 ОТВЕТ")
    print("=" * 70)
    
    # Печатаем ответ с отступами для читаемости
    answer_lines = result['answer'].split('\n')
    for line in answer_lines:
        print(f"  {line}")
    
    # Цитаты отдельно
    if result.get('quotes'):
        print("\n" + "=" * 70)
        print(" 📑 ИСТОЧНИКИ (цитаты из книг)")
        print("=" * 70)
        for i, quote in enumerate(result['quotes'], 1):
            print(f"  [{i}] {quote['source']}")
            # Форматируем цитату с отступом
            quote_text = quote['text']
            if len(quote_text) > 200:
                quote_text = quote_text[:200] + "..."
            print(f"      «{quote_text}»")
        print("\n" + "=" * 70)


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
                print("   Пример: «Что писала Татьяна Онегину»")
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
