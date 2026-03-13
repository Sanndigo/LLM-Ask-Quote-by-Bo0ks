# TXT Document Processing and Local LLM System

This project provides a complete solution for processing TXT files, creating semantic embeddings, and enabling local question-answering capabilities on the document content.

## Features

- **TXT File Processing**: Read and process TXT files from a specified directory
- **Text Chunking**: Split text into logical chunks using NLTK
- **Preprocessing**: Clean and preprocess text data
- **Semantic Embedding**: Generate embeddings using Sentence Transformers
- **FAISS Indexing**: Store and index embeddings for fast similarity search
- **Local LLM**: Answer questions based on document content without external APIs
- **Interactive Interfaces**: Both search and chat interfaces

## Architecture

```
data/
├── sample.txt          # Input TXT files
processed/
└── sample/
    └── sample_chunk_0.txt  # Processed text chunks
embeddings/
├── faiss_index.bin     # FAISS index file
└── faiss_index.bin_id_map.pkl  # ID mapping file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data:
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Basic Processing
```bash
python main_processor.py --step all
```

### Search Interface
```bash
python search_engine.py
```

For interactive search:
```bash
python search_engine.py --interactive
```

### Local LLM Chat
```bash
python local_llm.py
```

For interactive chat:
```bash
python local_llm.py --interactive
```

## Components

### 1. Text Processor (`text_processor.py`)
- Reads TXT files
- Preprocesses text
- Splits text into logical chunks
- Uses NLTK for sentence tokenization

### 2. Embedding Processor (`embedding_processor.py`)
- Creates semantic embeddings using Sentence Transformers
- Stores embeddings in FAISS index
- Handles indexing and retrieval

### 3. Main Processor (`main_processor.py`)
- Orchestrates the complete workflow
- Processes files in a directory
- Generates embeddings and indexes them

### 4. Search Engine (`search_engine.py`)
- Performs semantic search on indexed texts
- Finds relevant fragments based on user queries
- Provides interactive search interface
- Returns results with similarity scores

### 5. Local LLM (`local_llm.py`)
- Local language model for answering questions
- Works in context of uploaded documents
- Provides interactive chat interface
- Generates responses based on document content

## How It Works

1. **Document Processing**: TXT files are read and split into logical chunks
2. **Preprocessing**: Text is cleaned and normalized
3. **Chunking**: Text is divided into overlapping chunks for better semantic coverage
4. **Embedding Generation**: Each chunk is converted to a semantic embedding
5. **Indexing**: Embeddings are stored in FAISS index for fast similarity search
6. **Question Answering**: Local LLM uses document context to answer questions

## Example Workflow

1. Place TXT files in `data/` directory
2. Run processing:
   ```bash
   python main_processor.py --step all
   ```
3. Search documents:
   ```bash
   python search_engine.py --interactive
   ```
4. Ask questions about documents:
   ```bash
   python local_llm.py --interactive
   ```

## Requirements

- Python 3.7+
- NLTK
- Sentence Transformers
- FAISS
- NumPy
- Transformers
- PyTorch
- Accelerate
- Bitsandbytes

## License

MIT