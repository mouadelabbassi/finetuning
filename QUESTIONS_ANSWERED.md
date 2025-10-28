# ENSAKH RAG System - Questions Answered

## Your Questions

You asked three key questions about the RAG system code:
1. **What does this code do?**
2. **How to use it?**
3. **Can I run it independently?**

Here are the complete answers:

---

## 1. What Does This Code Do?

The ENSAKH RAG system is a **complete intelligent search and retrieval system** for an ENSAKH chatbot. It performs five main functions:

### A. Web Scraping (`ENSAKHScraper` class)
**Purpose**: Automatically collect information from the ENSAKH website

**What it does:**
- Visits multiple ENSAKH web pages (departments, programs, etc.)
- Extracts text content from HTML pages
- Downloads and processes PDF documents
- Implements smart caching to avoid re-downloading
- Respects rate limits (1 second between requests)
- Crawls up to 3 levels deep in the website

**Result**: A collection of documents with all ENSAKH information

### B. Text Processing (`TextProcessor` class)
**Purpose**: Prepare documents for efficient search

**What it does:**
- Splits long documents into chunks (800 words each, 100-word overlap)
- Ensures chunks aren't cut mid-sentence
- Normalizes queries (lowercase, expand abbreviations)
- Supports multilingual abbreviations:
  - "GI" → "génie informatique"
  - "IID" → "informatique ingénierie données"
  - "chkon" → "qui est who is"
  - "kifach" → "comment how"

**Result**: Clean, searchable text chunks with metadata

### C. Vector Storage (`VectorStore` class)
**Purpose**: Store and search documents efficiently

**What it does:**
- Converts text to mathematical vectors (embeddings) using AI models
- Stores vectors in ChromaDB (a specialized database)
- Implements two search methods:
  1. **Semantic search**: Finds documents by meaning (uses embeddings)
  2. **Keyword search**: Finds documents by exact words (uses BM25)
- Combines both methods for better results (hybrid search)
- Retrieves top 5 most relevant documents

**Result**: Fast, accurate document retrieval

### D. RAG Orchestration (`ENSAKHRag` class)
**Purpose**: Tie everything together

**What it does:**
- Manages the complete workflow
- Builds and updates knowledge base
- Retrieves relevant context for any query
- Formats context with sources and URLs
- Generates structured prompts for LLaMA models
- Includes strict rules to prevent AI hallucination

**Result**: Ready-to-use system for question answering

### E. Command-Line Interface (CLI)
**Purpose**: Easy interaction with the system

**What it provides:**
- `--build`: Build/update knowledge base
- `--test`: Run test queries
- `--rescrape`: Force fresh data scraping
- Progress bars and detailed logging

**Result**: User-friendly operation

---

## 2. How to Use It?

### Method 1: Command Line (Simplest)

**Installation:**
```bash
# Install all required packages
pip install -r requirements-rag.txt
```

**Build Knowledge Base (First Time):**
```bash
# This scrapes ENSAKH website and creates searchable database
# Takes 10-30 minutes on first run
python ensakh_rag.py --build
```

**Test the System:**
```bash
# Run pre-defined test queries
python ensakh_rag.py --test
```

**Update Data:**
```bash
# Get fresh data from website
python ensakh_rag.py --build --rescrape
```

### Method 2: Python Script (Basic)

```python
from ensakh_rag import ENSAKHRag

# Step 1: Initialize
rag = ENSAKHRag()

# Step 2: Build knowledge base (first time only)
rag.build_knowledge_base()

# Step 3: Ask questions
query = "what is IID?"
context, results = rag.retrieve(query)

# Step 4: See results
for i, result in enumerate(results, 1):
    print(f"[{i}] {result['metadata']['source']}")
    print(f"    {result['text'][:150]}...")
    print()
```

### Method 3: With LLaMA Model (Complete Chatbot)

```python
from ensakh_rag import ENSAKHRag
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize RAG
rag = ENSAKHRag()
rag.build_knowledge_base()

# Load your fine-tuned LLaMA model
model_path = "path/to/your/finetuned-llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create chatbot function
def ask_ensakh_chatbot(question):
    """Complete Q&A with context retrieval and LLaMA generation"""
    
    # Get relevant context from RAG
    context, sources = rag.retrieve(question)
    
    # Generate prompt with strict rules
    prompt = rag.generate_prompt(question, context)
    
    # Generate answer with LLaMA
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'answer': answer,
        'sources': [s['metadata']['url'] for s in sources]
    }

# Use it
response = ask_ensakh_chatbot("What is IID?")
print("Answer:", response['answer'])
print("Sources:", response['sources'])
```

### Method 4: As Web API

```python
from flask import Flask, request, jsonify
from ensakh_rag import ENSAKHRag

app = Flask(__name__)
rag = ENSAKHRag()
rag.build_knowledge_base()

@app.route('/ask', methods=['POST'])
def ask_question():
    """API endpoint for questions"""
    question = request.json.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Retrieve context
    context, results = rag.retrieve(question)
    
    # Return context and sources
    return jsonify({
        'context': context,
        'results': len(results),
        'sources': [
            {
                'text': r['text'][:200],
                'url': r['metadata']['url'],
                'score': r['score']
            }
            for r in results
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Use the API:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is IID?"}'
```

### Supported Query Languages

The system understands:

**English:**
- "what is IID?"
- "how can I apply to ENSAKH?"
- "who is the director?"

**French:**
- "c'est quoi le génie informatique?"
- "comment s'inscrire?"
- "quels sont les départements?"

**Darija (Moroccan Arabic):**
- "chkon hafidi?" (who is hafidi?)
- "kifach ndkhol?" (how to enter?)
- "3afak chno howa IID?" (please what is IID?)

---

## 3. Can I Run It Independently?

### Short Answer: **YES** ✅ (with one caveat)

### Detailed Answer:

#### What Works **WITHOUT** Any External Models:

✅ **1. Web Scraping**
```bash
python ensakh_rag.py --build
```
- Scrapes ENSAKH website
- No external dependencies needed
- Works completely standalone

✅ **2. Document Processing**
```python
from ensakh_rag import TextProcessor

processor = TextProcessor()
chunks = processor.chunk_text("long text here...")
```
- Splits documents into chunks
- No external models needed

✅ **3. Knowledge Base Building**
```bash
python ensakh_rag.py --build
```
- Creates vector database
- Downloads embedding model automatically
- Stores everything locally

✅ **4. Information Retrieval**
```python
from ensakh_rag import ENSAKHRag

rag = ENSAKHRag()
rag.build_knowledge_base()

# Returns relevant documents
context, results = rag.retrieve("what is IID?")
print(results)  # Shows documents, sources, scores
```
- **Works 100% independently**
- No LLaMA needed
- Returns actual document chunks

✅ **5. Prompt Generation**
```python
prompt = rag.generate_prompt(query, context)
print(prompt)  # Shows complete prompt
```
- Creates structured prompt
- No model needed to generate prompt
- Prompt is ready for any LLM

✅ **6. Testing**
```bash
python ensakh_rag.py --test
```
- Tests retrieval quality
- Shows what documents are found
- Validates the system works

#### What Needs LLaMA Model:

❌ **Generating Final Human Answers**
- The RAG system retrieves relevant information
- It creates prompts with that information
- But it doesn't generate the final conversational answer
- For that, you need to feed the prompt to LLaMA

**Comparison:**

| Task | Independent? | Output |
|------|-------------|--------|
| Scrape website | ✅ YES | Raw documents |
| Build knowledge base | ✅ YES | Vector database |
| Search for info | ✅ YES | Relevant document chunks |
| Generate prompt | ✅ YES | Formatted prompt string |
| Generate answer | ❌ NEEDS LLAMA | Human-readable answer |

### Use Cases Without LLaMA:

**1. Document Search System**
```python
# Find relevant documents about a topic
results = rag.retrieve("admission requirements")
for r in results:
    print(f"Source: {r['metadata']['url']}")
    print(f"Content: {r['text']}")
```

**2. Context Provider for Any LLM**
```python
# Use with GPT-4, Claude, or any other model
context, _ = rag.retrieve("what is IID?")
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": context}]
)
```

**3. Data Collection Tool**
```python
# Just use it to gather and organize ENSAKH data
scraper = ENSAKHScraper()
documents = scraper.scrape_all()
# Save or process documents as needed
```

**4. Research Tool**
```python
# Find all mentions of a topic
results = rag.retrieve("génie informatique")
# Analyze what documents say about the topic
```

### Complete Independence Summary:

```
┌─────────────────────────────────────────┐
│   Can Run 100% Independently:           │
│   - Scraping                             │  ← No external services
│   - Processing                           │  ← No external services
│   - Indexing                             │  ← No external services
│   - Searching                            │  ← No external services
│   - Prompt generation                    │  ← No external services
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│   Needs LLaMA (Optional):               │
│   - Generating human-like answers       │  ← Only for final Q&A
└─────────────────────────────────────────┘
```

**Bottom Line**: The RAG system is **fully functional independently**. It finds and retrieves information perfectly on its own. You only need LLaMA if you want it to generate conversational answers instead of just showing you the relevant documents.

---

## Quick Reference

### Installation
```bash
pip install -r requirements-rag.txt
python ensakh_rag.py --build  # Takes 10-30 min first time
```

### Basic Usage
```python
from ensakh_rag import ENSAKHRag

rag = ENSAKHRag()
rag.build_knowledge_base()

# Retrieve documents
context, results = rag.retrieve("your question")

# Generate prompt for LLaMA (optional)
prompt = rag.generate_prompt("your question", context)
```

### Independence
- ✅ Runs independently for retrieval
- ⚠️ Needs LLaMA only for final answer generation
- ✅ Can be used as standalone search engine
- ✅ Works with any LLM (GPT, Claude, LLaMA, etc.)

### Files Structure
```
finetuning/
├── ensakh_rag.py              # Main system (use this)
├── requirements-rag.txt       # Install dependencies
├── RAG_DOCUMENTATION.md       # Full technical docs
├── RAG_USAGE_GUIDE.md        # Quick start guide
└── README.md                 # Overview
```

### Need More Help?
- Technical details: `RAG_DOCUMENTATION.md`
- Examples: `RAG_USAGE_GUIDE.md`
- Quick overview: `README.md`

---

**SUMMARY**: The RAG system is a complete, independent information retrieval system that scrapes, processes, and searches ENSAKH data. It works perfectly on its own for finding relevant information. LLaMA is only needed if you want conversational AI responses instead of document retrieval results.
