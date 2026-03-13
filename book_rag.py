#!/usr/bin/env python3
"""
RAG-система для работы с книгами
Поиск фрагментов и ответы на вопросы с цитатами
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

# Настройка логирования
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BookRAG:
    """RAG-система для работы с книгами"""

    def __init__(
        self,
        index_path: str = 'embeddings/faiss_index.bin',
        id_map_path: str = 'embeddings/faiss_index.bin_id_map.pkl',
        model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        llm_model_name: str = 'Gemma-2-2b-it' 
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

        # Загрузка embedding модели
        self.embedder = SentenceTransformer(model_name)

        # Загрузка LLM
        self._load_llm_model()

    def _load_llm_model(self):
        """Загрузка LLM модели"""
        try:
            # Проверяем доступность GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🎮 Обнаружен GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Выбираем модель в зависимости от памяти GPU
                # GTX 1060 3GB = ~2.8GB доступно
                if gpu_memory >= 10:
                    self.llm_model_name = 'Qwen/Qwen2.5-3B-Instruct'
                    logger.info("✅ Достаточно памяти для 3B модели")
                elif gpu_memory >= 6:
                    self.llm_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
                    logger.info("✅ Загружаем 1.5B модель для GPU")
                elif gpu_memory >= 3:
                    self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
                    logger.info("✅ Загружаем 0.5B модель для GPU (оптимально для 3GB)")
                else:
                    logger.warning(f"⚠️ Мало видеопамяти ({gpu_memory:.1f}GB), пробуем CPU")
                    self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
            else:
                logger.info("GPU не обнаружен, используем CPU")
            
            logger.info(f"Загрузка LLM {self.llm_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
            
            # Загрузка модели
            if torch.cuda.is_available() and gpu_memory >= 3:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16,  # FP16 для GPU
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"  # Автоматически на GPU
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
            logger.warning(f"Не удалось загрузить {self.llm_model_name}: {e}")
            # Fallback
            try:
                self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.llm_model.eval()
                logger.info(f"✅ Загружена альтернативная модель: {self.llm_model_name}")
            except Exception as e2:
                logger.error(f"Не удалось загрузить LLM: {e2}")

    def load_index(self):
        """Загрузка индекса"""
        self.embedding_processor = EmbeddingProcessor(self.model_name)
        self.embedding_processor.load_index(self.index_path, self.id_map_path)
        self.loaded = True
        logger.info("Индекс загружен")

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
        
        if not fragments or fragments[0]['similarity'] < 0.3:
            return {
                'answer': "К сожалению, в загруженных текстах нет информации, которая могла бы ответить на ваш вопрос.",
                'quotes': [],
                'found': False
            }

        # Формируем контекст
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

        context_text = "\n\n".join(context_parts)

        # Если LLM не загружена - возвращаем только цитаты
        if self.llm_model is None or self.tokenizer is None:
            return {
                'answer': f"Найденные фрагменты по вашему запросу:\n\n{context_text}",
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

        user_prompt = f"""КОНТЕКСТ ИЗ КНИГ (используй ТОЛЬКО это):
{context_text}

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
