# ENSAKH RAG System Documentation

## Table of Contents
1. [Overview](#overview)
2. [What This Code Does](#what-this-code-does)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Can It Run Independently?](#can-it-run-independently)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The ENSAKH RAG (Retrieval Augmented Generation) System is a complete, production-ready implementation of an intelligent chatbot for the École Nationale des Sciences Appliquées de Khouribga (ENSAKH). It combines web scraping, document processing, semantic search, and prompt generation to provide accurate, context-aware responses about the school.

**Key Features:**
- 🕷️ **Intelligent Web Scraping**: Automatically extracts content from ENSAKH website
- 📄 **PDF Processing**: Extracts and processes information from PDF documents
- 🧠 **Multilingual Support**: Handles French, English, and Darija (Moroccan Arabic)
- 🔍 **Hybrid Search**: Combines semantic embeddings and keyword search (BM25)
- 💾 **Persistent Storage**: Uses ChromaDB for efficient vector storage
- 🚀 **Production Ready**: Includes caching, error handling, and rate limiting

---

## What This Code Does

### 1. **Web Scraping Module** (`ENSAKHScraper`)
- Scrapes the ENSAKH website (http://ensak.usms.ac.ma)
- Crawls multiple departments: Math & CS, Electrical Engineering, Networks & Telecoms, Process Engineering
- Extracts text content from web pages
- Downloads and processes PDF documents
- Implements intelligent caching to avoid redundant requests
- Respects rate limits and uses proper user agents

### 2. **Text Processing Module** (`TextProcessor`)
- Splits documents into manageable chunks (800 words with 100-word overlap)
- Normalizes queries by expanding abbreviations
- Supports multilingual query expansion (FR/EN/Darija)
- Examples:
  - "GI" → "génie informatique"
  - "chkon" → "qui est who is"
  - "kifach" → "comment how"

### 3. **Vector Store Module** (`VectorStore`)
- Uses sentence-transformers for multilingual embeddings
- Stores embeddings in ChromaDB (persistent vector database)
- Implements BM25 for keyword-based search
- Performs hybrid search combining:
  - Semantic similarity (using embeddings)
  - Keyword matching (using BM25)
- Returns top-k most relevant documents

### 4. **RAG Integration Module** (`ENSAKHRag`)
- Orchestrates all components
- Builds and maintains knowledge base
- Retrieves relevant context for queries
- Generates prompts formatted for LLaMA models
- Includes strict rules to prevent hallucination

### 5. **CLI & Testing**
- Command-line interface for building and testing
- Pre-defined test queries to validate the system
- Progress indicators and detailed logging

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Query Normalization                        │
│         (expand abbreviations, lowercase)               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Hybrid Search                              │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │ Semantic Search  │    │  Keyword Search  │          │
│  │   (Embeddings)   │    │     (BM25)       │          │
│  └────────┬─────────┘    └────────┬─────────┘          │
│           │                       │                     │
│           └───────────┬───────────┘                     │
│                       │                                 │
│              Merge & Rank Results                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Context Formatting                         │
│         (add sources, URLs, structure)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Prompt Generation                          │
│    (format for LLaMA with strict rules)                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            LLaMA Model (External)                       │
│         Generate final response                         │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for initial scraping and model downloads)

### Step-by-Step Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd finetuning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements-rag.txt
   ```

3. **First-time setup** (download models and build knowledge base):
   ```bash
   python ensakh_rag.py --build
   ```
   
   This will:
   - Download the multilingual embedding model (~500MB)
   - Scrape the ENSAKH website
   - Process and chunk documents
   - Create vector embeddings
   - Build the ChromaDB index
   
   **Note**: First run takes 10-30 minutes depending on internet speed and CPU.

---

## Usage Guide

### Command-Line Interface

#### 1. Build Knowledge Base (First Time or Update)
```bash
python ensakh_rag.py --build
```

#### 2. Force Rescrape (Get Fresh Data)
```bash
python ensakh_rag.py --build --rescrape
```

#### 3. Test the System
```bash
python ensakh_rag.py --test
```

This runs pre-defined test queries:
- "what is IID?"
- "who is hafidi?"
- "how can i enter to ensakh?"
- "génie informatique c'est quoi?"
- etc.

#### 4. Show Help
```bash
python ensakh_rag.py --help
```

### Programmatic Usage

#### Basic Example
```python
from ensakh_rag import ENSAKHRag

# Initialize the RAG system
rag = ENSAKHRag()

# Build knowledge base (only needed once)
rag.build_knowledge_base()

# Query the system
query = "what is IID?"
context, results = rag.retrieve(query)

# Generate prompt for LLaMA
prompt = rag.generate_prompt(query, context)

# Pass to your LLaMA model
# response = your_llama_model.generate(prompt)
print(prompt)
```

#### Advanced Example
```python
from ensakh_rag import ENSAKHRag, Config

# Customize configuration
Config.TOP_K = 10  # Return top 10 results
Config.CHUNK_SIZE = 1000  # Larger chunks

# Initialize
rag = ENSAKHRag()
rag.build_knowledge_base()

# Process multiple queries
queries = [
    "Quels sont les départements?",
    "Comment s'inscrire?",
    "What is the admission process?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    context, results = rag.retrieve(query)
    
    print(f"Found {len(results)} relevant documents:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.3f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Preview: {result['text'][:150]}...")
    
    # Generate and use prompt
    prompt = rag.generate_prompt(query, context)
    # ... send to LLaMA model
```

#### Integration with LLaMA Model
```python
from ensakh_rag import ENSAKHRag
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize RAG
rag = ENSAKHRag()
rag.build_knowledge_base()

# Load your fine-tuned LLaMA model
model_name = "path/to/your/finetuned-llama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Query function
def answer_question(question):
    # Get context from RAG
    context, results = rag.retrieve(question)
    prompt = rag.generate_prompt(question, context)
    
    # Generate response with LLaMA
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, results

# Example usage
question = "what is IID?"
answer, sources = answer_question(question)
print(f"Answer: {answer}")
print(f"\nSources: {[s['metadata']['url'] for s in sources]}")
```

---

## Can It Run Independently?

### ✅ YES - It Can Run Independently

The ENSAKH RAG system is **fully self-contained** and can run independently for:

1. **Knowledge Base Building**
   ```bash
   python ensakh_rag.py --build
   ```
   - Scrapes data
   - Processes documents
   - Creates embeddings
   - Builds vector database

2. **Information Retrieval**
   ```python
   from ensakh_rag import ENSAKHRag
   
   rag = ENSAKHRag()
   context, results = rag.retrieve("your query")
   ```
   - Returns relevant documents
   - Provides context with sources
   - Works without external models

3. **Testing**
   ```bash
   python ensakh_rag.py --test
   ```
   - Validates retrieval quality
   - Shows search results
   - No LLaMA model needed

### ⚠️ For Complete Q&A: Needs LLaMA Model

For generating **final answers** (not just retrieving context), you need:
- A fine-tuned LLaMA model
- The model trained on ENSAKH data (like in your dataset files)

**Workflow:**
```
RAG System (Independent) → Context Retrieval → Prompt Generation
                                                      ↓
                                            LLaMA Model (Separate)
                                                      ↓
                                              Final Answer
```

### What It Provides Without LLaMA
- ✅ Relevant document chunks
- ✅ Source URLs and metadata
- ✅ Structured context
- ✅ Ready-to-use prompts
- ❌ Final human-like answers (needs LLaMA)

### Standalone Use Cases
1. **Search Engine**: Use just retrieval for document search
2. **Context Provider**: Feed context to any LLM (GPT, Claude, etc.)
3. **Data Collection**: Scrape and organize ENSAKH data
4. **Testing Retrieval**: Validate search quality before model integration

---

## Configuration

Edit the `Config` class in `ensakh_rag.py`:

```python
@dataclass
class Config:
    # URLs to scrape (add/remove as needed)
    URLS = {
        'main': 'http://ensak.usms.ac.ma/ensak/',
        # ... add more
    }
    
    # Paths
    DATA_DIR = Path("./ensakh_data")  # Change storage location
    
    # Models
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Chunking
    CHUNK_SIZE = 800  # Increase for more context per chunk
    CHUNK_OVERLAP = 100  # Overlap between chunks
    
    # Retrieval
    TOP_K = 5  # Number of documents to retrieve
    SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
    
    # Scraping
    MAX_DEPTH = 3  # How deep to crawl links
    RATE_LIMIT = 1.0  # Seconds between requests
```

---

## API Reference

### `ENSAKHRag`
Main RAG system class.

#### Methods

**`build_knowledge_base(force_rescrape: bool = False)`**
- Builds or updates the knowledge base
- Parameters:
  - `force_rescrape`: If True, scrapes website even if cached data exists
- Returns: None
- Side effects: Creates/updates database files

**`retrieve(query: str) -> Tuple[str, List[Dict]]`**
- Retrieves relevant context for a query
- Parameters:
  - `query`: User question (any language)
- Returns:
  - `context`: Formatted text context for LLM
  - `results`: List of result dictionaries with metadata

**`generate_prompt(query: str, context: str) -> str`**
- Generates a prompt for LLaMA model
- Parameters:
  - `query`: User question
  - `context`: Retrieved context
- Returns: Complete prompt string

### `VectorStore`
Manages embeddings and vector database.

#### Methods

**`search(query: str, k: int = TOP_K) -> List[Dict]`**
- Performs hybrid search
- Returns: List of top-k documents with scores

### `TextProcessor`
Processes and chunks documents.

#### Methods

**`normalize_query(query: str) -> str`**
- Expands abbreviations and normalizes text
- Supports FR/EN/Darija

**`chunk_text(text: str) -> List[str]`**
- Splits text into overlapping chunks

### `ENSAKHScraper`
Web scraping functionality.

#### Methods

**`scrape_all() -> List[Dict]`**
- Scrapes all configured URLs
- Returns: List of document dictionaries

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'chromadb'
```
**Solution**: Install all dependencies
```bash
pip install -r requirements-rag.txt
```

#### 2. Model Download Issues
```
OSError: Can't load tokenizer for 'sentence-transformers/...'
```
**Solution**: Check internet connection, models download on first run

#### 3. Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU
```python
Config.BATCH_SIZE = 16  # Reduce from 32
```

#### 4. Scraping Fails
```
requests.exceptions.ConnectionError
```
**Solution**: 
- Check internet connection
- Verify ENSAKH website is accessible
- Increase timeout in `ENSAKHScraper._scrape_page()`

#### 5. Empty Results
**Symptoms**: No documents returned for queries
**Solutions**:
- Rebuild knowledge base: `python ensakh_rag.py --build --rescrape`
- Check if scraping succeeded (look for `ensakh_data/documents.json`)
- Verify ChromaDB created properly (`ensakh_data/chroma_db/`)

#### 6. Slow Performance
**Solutions**:
- Use cached data (don't use `--rescrape` frequently)
- Reduce `TOP_K` for faster searches
- Reduce `MAX_DEPTH` for faster scraping

### Getting Help

1. Check logs for detailed error messages
2. Verify all dependencies installed correctly
3. Ensure sufficient disk space (needs ~2GB for models + data)
4. Test with `--test` flag to validate setup

---

## File Structure

After running, you'll have:

```
finetuning/
├── ensakh_rag.py              # Main RAG system
├── requirements-rag.txt       # Dependencies
├── RAG_DOCUMENTATION.md       # This file
└── ensakh_data/               # Created on first run
    ├── documents.json         # Cached scraped documents
    ├── cache/                 # Page cache
    │   ├── *.json            # Cached page data
    │   └── temp.pdf          # Temporary PDF storage
    └── chroma_db/            # Vector database
        └── [ChromaDB files]
```

---

## Performance Metrics

- **Scraping**: 5-15 minutes (first time)
- **Building Knowledge Base**: 10-20 minutes (first time)
- **Subsequent Builds**: 2-5 minutes (uses cache)
- **Query Response Time**: 100-500ms
- **Embedding Model Size**: ~500MB
- **Database Size**: 100-500MB (depends on content)

---

## License & Credits

- **Author**: Claude (Anthropic)
- **Version**: 2.0
- **Date**: 2025
- **Purpose**: Educational - ENSAKH Chatbot

**Models Used**:
- Embeddings: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2

---

## Next Steps

1. **Basic Setup**: Install and build knowledge base
2. **Test Retrieval**: Run `--test` to verify it works
3. **Integrate with LLaMA**: Use your fine-tuned model from the dataset
4. **Customize**: Adjust configuration for your needs
5. **Deploy**: Use in production chatbot

For integration examples, see the [Usage Guide](#usage-guide) section above.
