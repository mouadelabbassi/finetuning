# ENSAKH RAG System - Quick Start Guide

## What is this?

The ENSAKH RAG (Retrieval Augmented Generation) System is an intelligent chatbot system for ENSAKH that:
- Scrapes and processes information from the ENSAKH website
- Stores it in a smart searchable database
- Retrieves relevant information for user questions
- Generates prompts for AI models to answer questions

## Quick Answer to Your Questions

### 1. What does this code do?

**In Simple Terms**: It's a smart search engine that helps a chatbot answer questions about ENSAKH accurately.

**Detailed Breakdown**:

1. **Scrapes Data** 🕷️
   - Visits ENSAKH website pages
   - Downloads PDFs
   - Extracts all text content
   - Saves it locally

2. **Processes Content** 📝
   - Breaks long documents into chunks
   - Cleans and organizes text
   - Expands abbreviations (GI → Génie Informatique)

3. **Creates Smart Index** 🧠
   - Converts text to mathematical vectors (embeddings)
   - Stores in a vector database (ChromaDB)
   - Enables semantic search (meaning-based, not just keywords)

4. **Retrieves Information** 🔍
   - Takes user questions
   - Finds most relevant content
   - Uses hybrid search (semantic + keyword matching)
   - Returns top results with sources

5. **Generates Prompts** 💬
   - Formats retrieved content
   - Creates structured prompts for LLaMA
   - Includes strict rules to prevent hallucination

### 2. How to use it?

#### Option A: Command Line (Easiest)

```bash
# Step 1: Install dependencies
pip install -r requirements-rag.txt

# Step 2: Build knowledge base (first time only, takes 10-30 min)
python ensakh_rag.py --build

# Step 3: Test it
python ensakh_rag.py --test
```

#### Option B: In Your Python Code

```python
from ensakh_rag import ENSAKHRag

# Initialize
rag = ENSAKHRag()

# Build knowledge base (first time only)
rag.build_knowledge_base()

# Use it
query = "what is IID?"
context, results = rag.retrieve(query)

# Generate prompt for your AI model
prompt = rag.generate_prompt(query, context)
print(prompt)
```

#### Option C: With Your LLaMA Model

```python
from ensakh_rag import ENSAKHRag

rag = ENSAKHRag()
rag.build_knowledge_base()

def ask_ensakh(question):
    # Get relevant context
    context, sources = rag.retrieve(question)
    
    # Generate prompt
    prompt = rag.generate_prompt(question, context)
    
    # Send to your LLaMA model
    response = your_llama_model.generate(prompt)
    
    return response

# Example
answer = ask_ensakh("What departments does ENSAKH have?")
print(answer)
```

### 3. Can it run independently?

**YES** ✅ - For Most Tasks:

**What Works Standalone:**
- ✅ Scraping ENSAKH website
- ✅ Building knowledge base
- ✅ Searching and retrieving information
- ✅ Generating prompts
- ✅ Testing the retrieval system

**What Needs External Model:**
- ❌ Generating final human-readable answers
- ❌ Complete question-answering (needs LLaMA)

**Think of it like this:**
```
┌─────────────────────────────────────────┐
│   ENSAKH RAG (This Code)                │
│   - Finds relevant information          │  ← Runs independently
│   - Prepares context                    │
│   - Creates prompts                     │
└────────────────┬────────────────────────┘
                 │ Passes prompt to...
                 ▼
┌─────────────────────────────────────────┐
│   LLaMA Model (Your Fine-tuned Model)   │  ← Separate component
│   - Reads the prompt                    │
│   - Generates human answer              │
└─────────────────────────────────────────┘
```

**Use Cases Without LLaMA:**
1. **Document Search**: Find relevant ENSAKH documents
2. **Data Collection**: Gather and organize ENSAKH info
3. **Context Provider**: Feed any AI model (GPT, Claude, etc.)
4. **Testing**: Validate search quality

**Use Cases With LLaMA:**
5. **Complete Chatbot**: Full question-answering system
6. **Production Deployment**: User-facing ENSAKH assistant

## Installation & Setup

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- Internet connection

### Step-by-Step Setup

```bash
# 1. Navigate to repository
cd finetuning

# 2. Install dependencies
pip install -r requirements-rag.txt

# 3. Build knowledge base (first time)
python ensakh_rag.py --build
```

**What happens during `--build`:**
1. Downloads embedding model (~500MB) - First time only
2. Scrapes ENSAKH website - Takes 5-15 minutes
3. Processes documents - Takes 2-5 minutes
4. Creates vector database - Takes 5-10 minutes
5. Total: **10-30 minutes** (first time only)

**Subsequent runs:**
- Uses cached data
- Much faster (2-5 minutes)
- Use `--rescrape` to get fresh data

## Common Usage Patterns

### Pattern 1: Simple Retrieval

```python
from ensakh_rag import ENSAKHRag

rag = ENSAKHRag()
rag.build_knowledge_base()

# Ask a question
context, results = rag.retrieve("How to enroll at ENSAKH?")

# See what was found
for i, result in enumerate(results, 1):
    print(f"\n[{i}] {result['metadata']['source']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...")
```

### Pattern 2: Batch Processing

```python
from ensakh_rag import ENSAKHRag

rag = ENSAKHRag()
rag.build_knowledge_base()

questions = [
    "What is IID?",
    "Who is the director?",
    "What are the admission requirements?"
]

for q in questions:
    context, results = rag.retrieve(q)
    print(f"\nQ: {q}")
    print(f"Found {len(results)} relevant documents")
```

### Pattern 3: Integration with Web API

```python
from flask import Flask, request, jsonify
from ensakh_rag import ENSAKHRag

app = Flask(__name__)
rag = ENSAKHRag()
rag.build_knowledge_base()

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    context, results = rag.retrieve(question)
    
    return jsonify({
        'context': context,
        'sources': [r['metadata']['url'] for r in results]
    })

if __name__ == '__main__':
    app.run(port=5000)
```

## Example Queries

The system understands multiple languages:

**English:**
- "what is IID?"
- "how can I enter ENSAKH?"
- "what departments does ENSAKH have?"

**French:**
- "c'est quoi le génie informatique?"
- "comment s'inscrire à l'ENSAKH?"
- "quels sont les départements?"

**Darija (Moroccan Arabic in Latin script):**
- "chkon hafidi?" (who is hafidi?)
- "kifach ndkhol l ensakh?" (how to enter ensakh?)
- "chno howa IID?" (what is IID?)

## Configuration

Edit `ensakh_rag.py` to customize:

```python
class Config:
    # Number of results to return
    TOP_K = 5  # Change to 10 for more results
    
    # Size of text chunks
    CHUNK_SIZE = 800  # Increase for more context
    
    # Scraping depth
    MAX_DEPTH = 3  # Lower for faster, higher for more complete
    
    # Storage location
    DATA_DIR = Path("./ensakh_data")  # Change path
```

## Testing

```bash
# Run built-in tests
python ensakh_rag.py --test
```

This will test with:
- "what is IID?"
- "who is hafidi?"
- "how can i enter to ensakh?"
- "génie informatique c'est quoi?"
- And more...

## Troubleshooting

### Problem: Module not found
```bash
# Solution: Install dependencies
pip install -r requirements-rag.txt
```

### Problem: Build takes too long
```bash
# Solution: Normal on first run (10-30 min)
# Subsequent builds are faster (uses cache)
```

### Problem: No results found
```bash
# Solution: Rebuild knowledge base
python ensakh_rag.py --build --rescrape
```

### Problem: Out of memory
```python
# Solution: Reduce batch size in code
# Edit Config class:
Config.BATCH_SIZE = 16  # Instead of 32
```

## File Structure

```
finetuning/
├── ensakh_rag.py              # Main system
├── requirements-rag.txt       # Dependencies
├── RAG_DOCUMENTATION.md       # Full documentation
├── RAG_USAGE_GUIDE.md        # This file
└── ensakh_data/              # Created on first run
    ├── documents.json        # Cached documents
    ├── cache/               # Scraped pages
    └── chroma_db/           # Vector database
```

## Performance

- **First Build**: 10-30 minutes
- **Subsequent Builds**: 2-5 minutes (cached)
- **Query Time**: 100-500ms
- **Memory Usage**: 1-2GB
- **Disk Space**: ~2GB (models + data)

## What's Next?

1. ✅ Run `python ensakh_rag.py --build`
2. ✅ Test with `python ensakh_rag.py --test`
3. ✅ Try your own queries in Python
4. ✅ Integrate with your LLaMA model
5. ✅ Deploy as chatbot

## Need Help?

- Read full documentation: `RAG_DOCUMENTATION.md`
- Check code comments in `ensakh_rag.py`
- Test individual components
- Verify internet connection for scraping

## Summary

**Can you run it independently?**
- ✅ YES - For retrieval and search
- ⚠️ PARTIAL - Needs LLaMA for complete Q&A

**How to use it?**
1. Install: `pip install -r requirements-rag.txt`
2. Build: `python ensakh_rag.py --build`
3. Use: Import and call `retrieve()` or `generate_prompt()`

**What does it do?**
- Scrapes ENSAKH website
- Creates searchable knowledge base
- Retrieves relevant information
- Prepares prompts for AI models
- Supports multiple languages
