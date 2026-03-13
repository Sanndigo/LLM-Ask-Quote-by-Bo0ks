#!/usr/bin/env python3
"""
Web-интерфейс для RAG-системы
"""
import os
import json
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from book_rag import BookRAG

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'data'
CORS(app)

# Глобальный экземпляр RAG
rag = None
rag_lock = threading.Lock()
indexing_in_progress = False


def get_rag():
    """Получение или создание экземпляра RAG"""
    global rag
    with rag_lock:
        if rag is None:
            rag = BookRAG()
            rag.load_index()
        return rag


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """Поиск фрагментов"""
    try:
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 5)
        
        if not query:
            return jsonify({'error': 'Введите запрос'}), 400
        
        rag = get_rag()
        fragments = rag.search_fragments(query, k=k)
        
        # Форматируем результат
        results = []
        for frag in fragments:
            results.append({
                'book': frag['source'],
                'position': frag.get('position', ''),
                'chapter': frag.get('chapter', ''),
                'similarity': round(frag['similarity'] * 100, 1),
                'content': frag['content']
            })

        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/answer', methods=['POST'])
def answer():
    """Ответ на вопрос"""
    try:
        data = request.json
        question = data.get('question', '')
        k = data.get('k', 5)
        
        if not question:
            return jsonify({'error': 'Введите вопрос'}), 400
        
        rag = get_rag()
        result = rag.answer(question, k=k)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result['answer'],
            'quotes': result['quotes'],
            'found': result['found']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/books', methods=['GET'])
def list_books():
    """Список загруженных книг"""
    books = []
    data_dir = app.config['UPLOAD_FOLDER']
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                size = os.path.getsize(filepath)
                books.append({
                    'name': filename,
                    'size': size,
                    'size_mb': round(size / 1024 / 1024, 2)
                })
    
    return jsonify({
        'success': True,
        'books': books,
        'count': len(books)
    })


@app.route('/api/upload', methods=['POST'])
def upload_book():
    """Загрузка новой книги"""
    global indexing_in_progress
    
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not file.filename.endswith('.txt'):
        return jsonify({'error': 'Только TXT файлы'}), 400
    
    # Сохраняем файл
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'message': f'Файл {filename} загружен',
        'filename': filename,
        'needs_indexing': True
    })


@app.route('/api/reindex', methods=['POST'])
def reindex():
    """Переиндексация книг"""
    global indexing_in_progress, rag
    
    if indexing_in_progress:
        return jsonify({'error': 'Индексация уже идет'}), 400
    
    def run_indexing():
        global indexing_in_progress, rag
        indexing_in_progress = True
        try:
            from main_processor import process_txt_files, create_embeddings
            
            # Обработка файлов
            process_txt_files('data', 'processed', chunk_size=256, overlap=64)
            
            # Создание эмбеддингов
            create_embeddings('processed', 'embeddings', 
                            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            
            # Перезагружаем RAG
            with rag_lock:
                rag = None  # Сброс для перезагрузки
            
        except Exception as e:
            print(f"Ошибка индексации: {e}")
        finally:
            indexing_in_progress = False
    
    # Запускаем в отдельном потоке
    thread = threading.Thread(target=run_indexing)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Индексация запущена в фоновом режиме'
    })


@app.route('/api/indexing-status', methods=['GET'])
def indexing_status():
    """Статус индексации"""
    return jsonify({
        'indexing': indexing_in_progress
    })


if __name__ == '__main__':
    # Создаем необходимые директории
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("=" * 60)
    print("🚀 Запуск веб-интерфейса RAG-системы")
    print("=" * 60)
    print("📍 Локальный доступ: http://localhost:5000")
    print("📍 Доступ из сети: http://192.168.1.34:5000")
    print("=" * 60)
    print("⚠️ Для доступа из сети:")
    print("   1. Открой порт 5000 в брандмауэре Windows")
    print("   2. Другие устройства: http://192.168.1.34:5000")
    print("=" * 60)
    
    # Запускаем на всех интерфейсах (0.0.0.0) для доступа из сети
    app.run(debug=False, host='0.0.0.0', port=5000)
