# ENSAKH RAG System 🎓🤖

A complete **Retrieval-Augmented Generation (RAG)** system for ENSAKH (École Nationale des Sciences Appliquées de Khouribga) that combines:
- **Web scraping** from ENSAKH website
- **Document processing** with semantic chunking
- **Vector embeddings** with ChromaDB
- **Fine-tuned LLAMA 3.1** model (elabbassimouad/LLAMA-ENSAKH)

## 🌟 Features

- ✅ **Multilingual Support**: English, French, and Arabic (Darija)
- ✅ **Smart Web Scraping**: Extracts clean content from ENSAKH website
- ✅ **Semantic Chunking**: Preserves context with intelligent text splitting
- ✅ **Deduplication**: Removes duplicate content automatically
- ✅ **Vector Search**: Fast similarity search with ChromaDB
- ✅ **Fine-tuned LLM**: Your custom LLAMA-ENSAKH model from HuggingFace
- ✅ **REST API**: FastAPI server for easy integration
- ✅ **Memory Efficient**: 4-bit quantization support

## 📁 Project Structure

```
rag_system/
├── web_scraper.py           # Scrapes ENSAKH website
├── document_processor.py    # Cleans and chunks documents
├── vector_store.py          # Creates embeddings & manages ChromaDB
├── rag_engine.py            # Main RAG engine (retrieval + generation)
├── build_knowledge_base.py  # Complete pipeline to build KB
├── api_server.py            # FastAPI REST API
├── test_rag.py              # Test script with examples
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install beautifulsoup4 requests chromadb sentence-transformers \
            transformers torch accelerate huggingface-hub lxml \
            html5lib pypdf python-docx fastapi uvicorn
```

### 2. Build Knowledge Base

This will scrape ENSAKH website, process documents, and create vector embeddings:

```bash
cd rag_system
python build_knowledge_base.py
```

**What it does:**
1. 📡 Scrapes ENSAKH website (50 pages, depth 2)
2. 🔧 Processes and chunks documents (512 chars, 50 overlap)
3. 🧠 Creates embeddings using multilingual model
4. 💾 Stores vectors in ChromaDB (`./chroma_db/`)

**Output:**
- `scraped_documents.json` - Raw scraped content
- `processed_chunks.json` - Cleaned and chunked text
- `chroma_db/` - Vector database

### 3. Test the RAG System

```bash
python test_rag.py
```

This will:
- Load your LLAMA-ENSAKH model from HuggingFace
- Test with sample questions
- Show retrieved context and generated answers

### 4. Start API Server

```bash
python api_server.py
```

Server runs on `http://localhost:8000`

**API Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `POST /query` - Query the RAG system
- `GET /stats` - System statistics
- `GET /docs` - Interactive API documentation

## 💻 Usage Examples

### Python API

```python
from rag_engine import ENSAKHRAGEngine
from vector_store import VectorStore

# Initialize
vector_store = VectorStore(collection_name="ensakh_knowledge")
rag_engine = ENSAKHRAGEngine(
    model_name="elabbassimouad/LLAMA-ENSAKH",
    vector_store=vector_store
)

# Query
result = rag_engine.query(
    question="What is Génie Informatique?",
    n_context_chunks=3,
    return_context=True
)

print(result['answer'])
```

### REST API

```bash
# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Génie Informatique at ENSAKH?",
    "n_context_chunks": 3,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "return_context": true
  }'
```

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "Comment s'inscrire à ENSAKH?",
        "n_context_chunks": 3,
        "return_context": True
    }
)

result = response.json()
print(result['answer'])
```

## 🔧 Configuration

### Web Scraper

```python
scraper = ENSAKHWebScraper(
    base_url="http://ensak.usms.ac.ma",
    max_depth=2  # How deep to crawl
)
documents = scraper.crawl(start_urls, max_pages=50)
```

### Document Processor

```python
processor = DocumentProcessor(
    chunk_size=512,      # Characters per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

### Vector Store

```python
vector_store = VectorStore(
    collection_name="ensakh_knowledge",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    persist_directory="./chroma_db"
)
```

### RAG Engine

```python
rag_engine = ENSAKHRAGEngine(
    model_name="elabbassimouad/LLAMA-ENSAKH",
    vector_store=vector_store,
    load_in_4bit=True  # Use 4-bit quantization
)
```

## 📊 How It Works

```
User Question
     ↓
1. Query Embedding (sentence-transformers)
     ↓
2. Vector Search (ChromaDB)
     ↓
3. Retrieve Top-K Relevant Chunks
     ↓
4. Format Prompt with Context
     ↓
5. Generate Answer (LLAMA-ENSAKH)
     ↓
Enhanced Answer
```

## 🎯 Key Components

### 1. Web Scraper (`web_scraper.py`)
- Fetches HTML from ENSAKH website
- Extracts main content (removes navigation, ads)
- Follows internal links intelligently
- Respects rate limiting

### 2. Document Processor (`document_processor.py`)
- Cleans text (removes noise, normalizes)
- Semantic chunking (preserves context)
- Deduplication (hash-based)
- Quality filtering

### 3. Vector Store (`vector_store.py`)
- Creates embeddings (multilingual model)
- Stores in ChromaDB
- Fast similarity search
- Persistent storage

### 4. RAG Engine (`rag_engine.py`)
- Retrieves relevant context
- Formats prompts for LLAMA
- Generates answers
- Complete pipeline

## 🌐 Supported Languages

The system supports **multilingual** queries and responses:
- 🇬🇧 **English**: "What is Génie Informatique?"
- 🇫🇷 **French**: "Qu'est-ce que le Génie Informatique?"
- 🇲🇦 **Darija**: "شنو هو الجيني انفورماتيك؟"

## 📈 Performance

- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- **LLM**: LLAMA 3.1 8B (4-bit quantized)
- **Memory**: ~6-8GB GPU RAM (with 4-bit quantization)
- **Speed**: ~2-5 seconds per query (depends on hardware)

## 🔒 Security Notes

- API runs on `0.0.0.0:8000` by default
- In production, configure CORS properly
- Add authentication if needed
- Use HTTPS in production

## 🐛 Troubleshooting

### "No module named 'chromadb'"
```bash
pip install chromadb
```

### "CUDA out of memory"
- Use 4-bit quantization: `load_in_4bit=True`
- Reduce `max_new_tokens`
- Reduce `n_context_chunks`

### "Collection not found"
```bash
# Rebuild knowledge base
python build_knowledge_base.py
```

### Model loading issues
```bash
# Login to HuggingFace
huggingface-cli login
```

## 📝 Customization

### Add More URLs

Edit `build_knowledge_base.py`:

```python
start_urls = [
    "http://ensak.usms.ac.ma/ensak/",
    "http://ensak.usms.ac.ma/ensak/your-new-page/",
    # Add more URLs
]
```

### Change Embedding Model

Edit `vector_store.py`:

```python
embedding_model = "sentence-transformers/your-model-name"
```

### Adjust Chunk Size

Edit `build_knowledge_base.py`:

```python
chunk_size = 1024  # Larger chunks
chunk_overlap = 100
```

## 🎓 Example Queries

```python
# Academic questions
"What programs does ENSAKH offer?"
"Quels sont les programmes de formation?"

# Admission questions
"How do I apply to ENSAKH?"
"Comment s'inscrire à ENSAKH?"

# Technical questions
"What is machine learning?"
"Explain neural networks"

# Department questions
"Tell me about Génie Informatique"
"Qu'est-ce que le département GI?"
```

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "api_server.py"]
```

### Cloud Deployment

- **AWS**: EC2 with GPU (g4dn.xlarge)
- **Google Cloud**: Compute Engine with GPU
- **Azure**: VM with GPU
- **HuggingFace Spaces**: Gradio interface

## 📚 References

- [LLAMA 3.1 Documentation](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

Feel free to:
- Add more data sources
- Improve chunking strategies
- Optimize embeddings
- Enhance the API

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Built for ENSAKH with ❤️

---

**Need help?** Check the logs or open an issue!
