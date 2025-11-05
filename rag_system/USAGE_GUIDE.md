# ENSAKH RAG System - Usage Guide 📖

## 🎯 Overview

This RAG (Retrieval-Augmented Generation) system enhances your fine-tuned LLAMA-ENSAKH model with real-time knowledge from the ENSAKH website.

**How it works:**
```
User Question → Vector Search → Retrieve Context → LLM Generation → Enhanced Answer
```

## 🚀 Getting Started (3 Steps)

### Step 1: Install Dependencies

```bash
cd rag_system
pip install -r requirements.txt
```

### Step 2: Build Knowledge Base

```bash
python build_knowledge_base.py
```

This will:
- Scrape ENSAKH website (~5-10 minutes)
- Process and chunk documents
- Create embeddings
- Store in ChromaDB

### Step 3: Test the System

```bash
python quick_start.py
```

## 📚 Detailed Usage

### Option A: Python API (Recommended for Integration)

```python
from rag_engine import ENSAKHRAGEngine
from vector_store import VectorStore

# Initialize once
vector_store = VectorStore(collection_name="ensakh_knowledge")
rag = ENSAKHRAGEngine(
    model_name="elabbassimouad/LLAMA-ENSAKH",
    vector_store=vector_store
)

# Query multiple times
result = rag.query("What is Génie Informatique?")
print(result['answer'])
```

### Option B: REST API (Recommended for Web Apps)

**Start server:**
```bash
python api_server.py
```

**Query from any language:**

```python
# Python
import requests
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is ENSAKH?"}
)
print(response.json()['answer'])
```

```javascript
// JavaScript
fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({question: 'What is ENSAKH?'})
})
.then(r => r.json())
.then(data => console.log(data.answer));
```

```bash
# cURL
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ENSAKH?"}'
```

### Option C: Interactive Mode

```bash
python test_rag.py interactive
```

Type questions and get instant answers!

## 🎨 Advanced Usage

### Custom Parameters

```python
result = rag.query(
    question="Explain machine learning",
    n_context_chunks=5,      # More context
    max_new_tokens=1000,     # Longer answer
    temperature=0.5,         # More focused
    return_context=True      # See retrieved docs
)

# Access retrieved context
for chunk in result['context']:
    print(f"Source: {chunk['metadata']['title']}")
    print(f"Relevance: {chunk['relevance']:.2%}")
```

### Filter by Source

```python
# Search only in specific documents
results = vector_store.search(
    query="admission",
    n_results=5,
    filter_metadata={"title": "Formation Initiale"}
)
```

### Batch Processing

```python
questions = [
    "What is GI?",
    "What is IID?",
    "What is GPEE?"
]

for q in questions:
    result = rag.query(q)
    print(f"Q: {q}\nA: {result['answer']}\n")
```

## 🔧 Configuration

### Adjust Retrieval

```python
# More context = better answers but slower
result = rag.query(question, n_context_chunks=5)

# Less context = faster but may miss info
result = rag.query(question, n_context_chunks=1)
```

### Adjust Generation

```python
# More creative (higher temperature)
result = rag.query(question, temperature=0.9)

# More focused (lower temperature)
result = rag.query(question, temperature=0.3)

# Longer answers
result = rag.query(question, max_new_tokens=1000)
```

### Memory Optimization

```python
# Use 4-bit quantization (saves ~75% memory)
rag = ENSAKHRAGEngine(
    model_name="elabbassimouad/LLAMA-ENSAKH",
    vector_store=vector_store,
    load_in_4bit=True  # ← Enable this
)
```

## 📊 Monitoring

### Check System Health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "vector_store_loaded": true,
  "collection_stats": {
    "total_chunks": 245
  }
}
```

### Get Statistics

```python
stats = vector_store.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

## 🐛 Troubleshooting

### Issue: "Collection not found"

**Solution:**
```bash
python build_knowledge_base.py
```

### Issue: "CUDA out of memory"

**Solution 1:** Use 4-bit quantization
```python
rag = ENSAKHRAGEngine(..., load_in_4bit=True)
```

**Solution 2:** Reduce parameters
```python
result = rag.query(
    question,
    n_context_chunks=2,    # Reduce from 3
    max_new_tokens=256     # Reduce from 512
)
```

### Issue: "Model not found on HuggingFace"

**Solution:** Login to HuggingFace
```bash
pip install huggingface-hub
huggingface-cli login
```

### Issue: Slow responses

**Causes & Solutions:**
- **First query is slow:** Model loading (normal)
- **All queries slow:** Reduce `max_new_tokens`
- **Retrieval slow:** Reduce `n_context_chunks`

## 🎯 Best Practices

### 1. Question Formulation

✅ **Good:**
- "What is Génie Informatique at ENSAKH?"
- "How do I apply to ENSAKH?"
- "Explain the admission process"

❌ **Avoid:**
- "GI?" (too vague)
- Very long questions (>200 words)

### 2. Context Size

- **General questions:** 2-3 chunks
- **Detailed questions:** 4-5 chunks
- **Simple facts:** 1-2 chunks

### 3. Temperature Settings

- **Factual answers:** 0.3-0.5
- **Explanations:** 0.6-0.8
- **Creative content:** 0.8-1.0

## 📈 Performance Tips

### Speed Optimization

```python
# 1. Use 4-bit quantization
load_in_4bit=True

# 2. Reduce context
n_context_chunks=2

# 3. Limit output length
max_new_tokens=256

# 4. Batch similar queries
# (reuses loaded model)
```

### Quality Optimization

```python
# 1. More context
n_context_chunks=5

# 2. Lower temperature
temperature=0.5

# 3. Return context for verification
return_context=True
```

## 🔄 Updating Knowledge Base

### Add New Documents

```python
from web_scraper import ENSAKHWebScraper
from document_processor import DocumentProcessor
from vector_store import VectorStore

# Scrape new URLs
scraper = ENSAKHWebScraper()
new_docs = scraper.crawl(["http://ensak.usms.ac.ma/new-page/"])

# Process
processor = DocumentProcessor()
chunks = processor.process_documents(new_docs)

# Add to existing store
vector_store = VectorStore(collection_name="ensakh_knowledge")
vector_store.add_documents(chunks)
```

### Rebuild from Scratch

```bash
# Delete old database
rm -rf chroma_db/

# Rebuild
python build_knowledge_base.py
```

## 🌐 Deployment

### Local Development

```bash
python api_server.py
# Access at http://localhost:8000
```

### Production Deployment

```bash
# Use production server
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "api_server.py"]
```

```bash
docker build -t ensakh-rag .
docker run -p 8000:8000 ensakh-rag
```

## 📞 API Reference

### POST /query

**Request:**
```json
{
  "question": "What is ENSAKH?",
  "n_context_chunks": 3,
  "max_new_tokens": 512,
  "temperature": 0.7,
  "return_context": false
}
```

**Response:**
```json
{
  "question": "What is ENSAKH?",
  "answer": "ENSAKH is...",
  "context": [...]  // if return_context=true
}
```

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vector_store_loaded": true,
  "collection_stats": {
    "collection_name": "ensakh_knowledge",
    "total_chunks": 245
  }
}
```

### GET /stats

**Response:**
```json
{
  "model": "elabbassimouad/LLAMA-ENSAKH",
  "vector_store": {
    "collection_name": "ensakh_knowledge",
    "total_chunks": 245,
    "persist_directory": "./chroma_db"
  }
}
```

## 🎓 Example Use Cases

### 1. Student Chatbot

```python
def student_assistant(question):
    result = rag.query(
        question=question,
        n_context_chunks=3,
        temperature=0.7
    )
    return result['answer']

# Usage
answer = student_assistant("How do I apply?")
```

### 2. FAQ System

```python
faqs = [
    "What programs does ENSAKH offer?",
    "What are the admission requirements?",
    "How much are the tuition fees?"
]

for faq in faqs:
    result = rag.query(faq, temperature=0.5)
    print(f"Q: {faq}")
    print(f"A: {result['answer']}\n")
```

### 3. Multi-language Support

```python
questions = {
    "en": "What is Génie Informatique?",
    "fr": "Qu'est-ce que le Génie Informatique?",
    "ar": "ما هو الجيني انفورماتيك؟"
}

for lang, q in questions.items():
    result = rag.query(q)
    print(f"[{lang}] {result['answer']}\n")
```

## 💡 Tips & Tricks

1. **Cache frequently asked questions** for faster responses
2. **Use lower temperature (0.3-0.5)** for factual questions
3. **Return context** during development to verify retrieval
4. **Monitor token usage** to optimize costs
5. **Update knowledge base** regularly for fresh content

## 🆘 Getting Help

- Check logs: `tail -f api_server.log`
- Test retrieval only: `python test_rag.py search`
- Interactive debugging: `python test_rag.py interactive`
- Check documentation: `http://localhost:8000/docs`

---

**Happy RAG-ing! 🚀**
