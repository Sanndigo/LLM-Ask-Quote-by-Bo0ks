# 📚 RAG - Book Search with AI Answers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial](https://img.shields.io/badge/license-NC--BY-red.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v1.1.0-green.svg)](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks)

---

**🌐 Language / Язык:** 
- **[🇬🇧 English](README.en.md)** (current)
- **[🇷🇺 Русский](README.md)**

---

Intelligent book search and Q&A system using **semantic chunking** and **Mistral AI API**.

## ✨ Features

### 🔍 Smart Search with Semantic Chunking
- **4764 chunks** with smart boundaries (sentence-by-sentence)
- **Never cuts words** - preserves sentence integrity
- **Semantic grouping** - combines by meaning
- Shows 5 most relevant fragments
- **Displays chapters, verses, parts** when found

### 💬 Context Popup
- **Click on result** - opens popup with context
- **Shows neighboring chunks** (2 on each side)
- **Full text** without truncation

### ❓ AI-Powered Q&A with Mistral AI
- AI generates detailed answers in Russian/English
- **Uses context from found fragments**
- **No hallucinations** - answers only from context
- Shows quotes from books

### 📖 Library Management
- Modern web interface
- TXT file upload
- Automatic indexing of new books
- Support for Russian and English texts

## ❗ Important

- Ask **detailed questions** - AI understands context better
- System answers **ONLY from uploaded books**
- If book is missing - upload it to `data/` folder

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Mistral AI API

**Get API Key:**
1. Go to https://console.mistral.ai/
2. Register
3. Go to **API Keys**
4. Create key

**Configure .env:**
```bash
# Create .env file in project root
echo MISTRAL_API_KEY=your_key > .env
```

**Free:** 30 days trial, then ~$0.15 per 1M tokens

### 3. Index Books

```bash
# Re-index (if books changed)
python main_processor.py --step all --threshold 0.45
```

### 4. Run Web Interface

```bash
python web_app.py
```

Open in browser: **http://localhost:5000**

### 5. Console Mode (Optional)

```bash
python book_rag.py
```

## 📋 Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4 GB | 8 GB |
| **Storage** | 2 GB | 6 GB |
| **GPU** | Not required | Any (optional) |

### For Mistral AI API:
- Internet connection
- Mistral AI API key
- **Free:** 30 days trial

## 📁 Project Structure

```
TechnoStr/
├── web_app.py              # Flask web server
├── book_rag.py             # RAG engine with Mistral AI
├── semantic_processor.py   # Semantic chunking
├── embedding_processor.py  # Vector search (FAISS)
├── text_processor.py       # Text processing
├── main_processor.py       # Book processing
├── search_engine.py        # Search (legacy)
├── templates/
│   └── index.html          # Web interface
├── data/                   # TXT books
├── processed/              # Processed chunks (don't commit)
└── embeddings/             # Vector index (don't commit)
```

## 🎯 Usage Examples

### Search Fragment

**Query:** "where does Tatyana write letter to Onegin"

**Result:**
```
📖 Eugene Onegin
📍 Stanza XXV | Fragment #152
📊 Similarity: 85.3%
📝 But having received Tati...
🔍 Click to view context
```

**Click** → popup opens with neighboring chunks!

### Answer Question

**Question:** "Who wrote The Overcoat?"

**Answer:**
```
According to the text from "The Overcoat", the author is
Nikolai Vasilievich Gogol. The work was written in 1842
and tells about the clerk Akaky...

📑 QUOTES (sources):
1. The Overcoat (fragment #1)
   «Nikolai Vasilievich Gogol The Overcoat...»
```

## 💰 Mistral AI Pricing

| Tier | Limit | Price |
|------|-------|-------|
| **Trial** | 30 days | 0 € |
| **Pay-as-you-go** | Per use | ~€0.15 per 1M tokens |

After trial - pay per use. Free tier is enough for testing!

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Mistral AI API
MISTRAL_API_KEY=your_key
```

### Book Processing Parameters

```bash
# Process with custom parameters
python main_processor.py --step all --threshold 0.45
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.45 | Similarity threshold for chunking (0-1) |
| `--step` | all | Stage: process/embed/all |

## 🐛 Troubleshooting

### "Mistral API error: 401"
- Check API key in `.env`
- Make sure key is active
- Check if trial period expired

### "Mistral API error: 429"
- Request limit exceeded
- Wait or upgrade plan

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install openai
```

### Popup not opening
- **Refresh page (F5)**
- Open browser console (F12) → check errors
- Make sure JavaScript is enabled

### Chunks cut mid-word
- **Re-index:**
  ```bash
  rmdir /s /q processed embeddings
  python main_processor.py --step all
  ```

## 📄 License

**Non-Commercial License (CC BY-NC 4.0 Compatible)**

**Copyright Holder:** Serafim Grekov (Sanndigo)

- ✅ **Allowed:** personal use, education, research
- ❌ **Prohibited:** commercial use, sale, monetization
- 📋 **Requirements:** attribution, preserve license

Valid in: **Russian Federation** and **USA**

See full text in [LICENSE](LICENSE) file

## 👥 Authors

- **[@Sanndigo](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks)** - Main development
- **Serafim Grekov** - Copyright holder

## 📬 Contacts

- **GitHub Issues:** [Report a problem](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks/issues)
- **Repository:** https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks

---

<div align="center">

**Made with ❤️ for book lovers**

[⭐ Star on GitHub](https://github.com/Sanndigo/LLM-Ask-Quote-by-Bo0ks/stargazers)

</div>
