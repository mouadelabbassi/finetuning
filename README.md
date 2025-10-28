# ENSAKH Finetuning & RAG System

This repository contains resources for building an intelligent ENSAKH chatbot:
- Fine-tuning datasets for LLaMA models
- Complete RAG (Retrieval Augmented Generation) system

## 📚 Contents

### 1. Fine-tuning Datasets
- `ensakh_llm_finetune_dataset.jsonl` - Main fine-tuning dataset
- `input.jsonl` - Raw input data
- `input_final_cleaned.jsonl` - Cleaned input data
- `aa.json` - Additional data
- `ENSAKH Dataset Augmentation & Technical Expansion.pdf` - Dataset documentation

### 2. RAG System (NEW ✨)
- `ensakh_rag.py` - Complete RAG implementation
- `RAG_DOCUMENTATION.md` - Full technical documentation
- `RAG_USAGE_GUIDE.md` - Quick start guide
- `requirements-rag.txt` - Python dependencies

## 🚀 Quick Start - RAG System

### What is the RAG System?
A complete Retrieval Augmented Generation system that:
- ✅ Scrapes ENSAKH website automatically
- ✅ Creates a smart searchable knowledge base
- ✅ Retrieves relevant information for questions
- ✅ Generates prompts for AI models
- ✅ Supports French, English, and Darija

### Installation

```bash
# Install dependencies
pip install -r requirements-rag.txt

# Build knowledge base (first time, takes 10-30 min)
python ensakh_rag.py --build

# Test the system
python ensakh_rag.py --test
```

### Basic Usage

```python
from ensakh_rag import ENSAKHRag

# Initialize
rag = ENSAKHRag()
rag.build_knowledge_base()

# Ask a question
query = "what is IID?"
context, results = rag.retrieve(query)

# Generate prompt for LLaMA
prompt = rag.generate_prompt(query, context)
print(prompt)
```

### Can It Run Independently?

**YES** ✅ - The RAG system runs independently for:
- Document scraping and indexing
- Information retrieval
- Prompt generation

**Needs LLaMA** for:
- Generating final human-readable answers
- Complete question-answering

See `RAG_USAGE_GUIDE.md` for detailed usage examples.

## 📖 Documentation

- **`RAG_DOCUMENTATION.md`** - Complete technical documentation
  - System architecture
  - API reference
  - Configuration options
  - Troubleshooting guide

- **`RAG_USAGE_GUIDE.md`** - Quick start guide
  - What it does
  - How to use it
  - Example code
  - Common patterns

## 🏗️ System Architecture

```
User Query → RAG System → Knowledge Base
                ↓
        Context Retrieval
                ↓
        Prompt Generation → LLaMA Model → Answer
```

## 🔧 Features

### RAG System Features
- 🕷️ **Web Scraping**: Automatic ENSAKH website crawling
- 📄 **PDF Processing**: Extract text from documents
- 🧠 **Multilingual**: French, English, Darija support
- 🔍 **Hybrid Search**: Semantic + keyword search
- 💾 **Persistent Storage**: ChromaDB vector database
- 🚀 **Production Ready**: Caching, error handling, rate limiting

### Dataset Features
- Comprehensive ENSAKH information
- Multiple formats (JSONL, JSON)
- Ready for fine-tuning
- Documented augmentation process

## 📊 Example Queries

The RAG system handles multiple languages:

**English:**
```python
rag.retrieve("what is IID?")
rag.retrieve("how can I enter ENSAKH?")
```

**French:**
```python
rag.retrieve("c'est quoi le génie informatique?")
rag.retrieve("comment s'inscrire?")
```

**Darija:**
```python
rag.retrieve("chkon hafidi?")
rag.retrieve("kifach ndkhol l ensakh?")
```

## 🛠️ Integration with LLaMA

```python
from ensakh_rag import ENSAKHRag

rag = ENSAKHRag()
rag.build_knowledge_base()

# Your fine-tuned LLaMA model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

def answer_question(question):
    # Get context from RAG
    context, sources = rag.retrieve(question)
    prompt = rag.generate_prompt(question, context)
    
    # Generate with LLaMA
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📦 Requirements

- Python 3.8+
- 4GB+ RAM
- Internet connection (for initial setup)

See `requirements-rag.txt` for Python packages.

## 🤝 Contributing

This is an educational project for ENSAKH. Feel free to:
- Report issues
- Suggest improvements
- Add more data sources
- Enhance the RAG system

## 📝 License

Educational purpose - ENSAKH Chatbot System

## 🔗 Related Files

- Fine-tuning datasets in repository root
- RAG system code: `ensakh_rag.py`
- Documentation: `RAG_DOCUMENTATION.md`, `RAG_USAGE_GUIDE.md`

---

**Get Started**: `python ensakh_rag.py --build`