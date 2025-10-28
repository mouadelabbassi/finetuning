"""
ENSAKH RAG System - Production Implementation
============================================
A complete Retrieval Augmented Generation system for ENSAKH chatbot

Author: Claude (Anthropic)
Version: 2.0
Date: 2025

Requirements:
    pip install chromadb sentence-transformers beautifulsoup4 requests
    pip install pypdf2 langchain tqdm python-dotenv rank-bm25
    pip install transformers torch unstructured
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
import hashlib

# Core libraries
import requests
from bs4 import BeautifulSoup
import PyPDF2
from tqdm import tqdm

# ML/NLP libraries
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch

# Configuration
@dataclass
class Config:
    """System configuration"""
    # URLs to scrape
    URLS = {
        'main': 'http://ensak.usms.ac.ma/ensak/',
        'departments': 'http://ensak.usms.ac.ma/ensak/departements/',
        'formation_continue': 'http://ensak.usms.ac.ma/ensak/formation-continue/',
        'formation_initiale': 'http://ensak.usms.ac.ma/ensak/formation-initiale/',
        'math_info': 'http://ensak.usms.ac.ma/ensak/maths-informatique/',
        'genie_electrique': 'http://ensak.usms.ac.ma/ensak/genie-electrique/',
        'genie_reseaux': 'http://ensak.usms.ac.ma/ensak/genie-reseaux-telecoms/',
        'genie_procedes': 'http://ensak.usms.ac.ma/ensak/genie-des-procedes/'
    }
    
    # Paths
    DATA_DIR = Path("./ensakh_data")
    CHROMA_DIR = DATA_DIR / "chroma_db"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Models
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    # Retrieval
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Scraping
    MAX_DEPTH = 3
    RATE_LIMIT = 1.0  # seconds between requests
    USER_AGENT = "ENSAKHBot/2.0 (Educational Purpose)"

config = Config()

# ============================================
# 1. WEB SCRAPER MODULE
# ============================================

class ENSAKHScraper:
    """Intelligent web scraper for ENSAKH website"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.USER_AGENT})
        self.visited_urls = set()
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def scrape_all(self) -> List[Dict]:
        """Scrape all configured URLs"""
        print("🕷️  Starting ENSAKH web scraping...")
        documents = []
        
        for name, url in tqdm(config.URLS.items(), desc="Scraping URLs"):
            try:
                docs = self.scrape_page(url, name, depth=0)
                documents.extend(docs)
                print(f"  ✓ {name}: {len(docs)} documents extracted")
            except Exception as e:
                print(f"  ✗ {name}: Error - {str(e)}")
        
        print(f"\n✅ Total documents scraped: {len(documents)}")
        return documents
    
    def scrape_page(self, url: str, source: str, depth: int = 0) -> List[Dict]:
        """Recursively scrape a page and its links"""
        if depth > config.MAX_DEPTH or url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        documents = []
        
        try:
            # Check cache first
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Fetch page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            content = self._extract_content(soup)
            if content:
                doc = {
                    'url': url,
                    'source': source,
                    'content': content,
                    'title': soup.find('title').text if soup.find('title') else '',
                    'metadata': {
                        'scraped_at': datetime.now().isoformat(),
                        'depth': depth
                    }
                }
                documents.append(doc)
            
            # Extract PDFs
            pdf_docs = self._extract_pdfs(soup, url, source)
            documents.extend(pdf_docs)
            
            # Follow internal links
            if depth < config.MAX_DEPTH:
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    if self._is_internal_link(next_url, url):
                        documents.extend(
                            self.scrape_page(next_url, source, depth + 1)
                        )
            
            # Cache results
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"    Error scraping {url}: {str(e)}")
        
        return documents
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_pdfs(self, soup: BeautifulSoup, base_url: str, source: str) -> List[Dict]:
        """Download and extract text from PDFs"""
        documents = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.pdf'):
                pdf_url = urljoin(base_url, href)
                try:
                    pdf_content = self._download_pdf(pdf_url)
                    if pdf_content:
                        documents.append({
                            'url': pdf_url,
                            'source': source,
                            'content': pdf_content,
                            'title': link.text.strip() or Path(pdf_url).name,
                            'metadata': {
                                'type': 'pdf',
                                'scraped_at': datetime.now().isoformat()
                            }
                        })
                except Exception as e:
                    print(f"    Error processing PDF {pdf_url}: {str(e)}")
        
        return documents
    
    def _download_pdf(self, url: str) -> Optional[str]:
        """Download and extract text from PDF"""
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        # Save temporarily
        temp_path = self.cache_dir / "temp.pdf"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Extract text
        text = ""
        with open(temp_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        temp_path.unlink()
        return text.strip()
    
    def _is_internal_link(self, url: str, base_url: str) -> bool:
        """Check if URL is internal to ENSAKH site"""
        return urlparse(url).netloc == urlparse(base_url).netloc


# ============================================
# 2. TEXT PROCESSING MODULE
# ============================================

class TextProcessor:
    """Process and chunk documents for embedding"""
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents into chunks"""
        print("\n📝 Processing documents...")
        
        processed = []
        for doc in tqdm(documents, desc="Processing"):
            chunks = self.chunk_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                processed.append({
                    'id': f"{hashlib.md5(doc['url'].encode()).hexdigest()}_{i}",
                    'text': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'source': doc['source'],
                        'url': doc['url'],
                        'title': doc.get('title', ''),
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                })
        
        print(f"✅ Created {len(processed)} chunks from {len(documents)} documents")
        return processed
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap // 10)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def normalize_query(self, query: str) -> str:
        """Normalize and expand query"""
        query = query.lower().strip()
        
        # Expand common abbreviations
        expansions = {
            'gi': 'génie informatique',
            'iid': 'informatique ingénierie données',
            'iric': 'ingénierie réseaux informatique communication',
            'gp': 'génie procédés',
            'ge': 'génie électrique',
            'api': 'automatique productique industrielle',
            'prof': 'professeur enseignant',
            'chkon': 'qui est who is',
            '3afak': 'sil vous plait please',
            'chno': 'quoi what',
            'chnou': 'quoi what',
            'kifach': 'comment how',
            'fach': 'quand when',
            'fin': 'où where'
        }
        
        for abbr, full in expansions.items():
            if abbr in query:
                query = query.replace(abbr, f"{abbr} {full}")
        
        return query


# ============================================
# 3. EMBEDDING & VECTOR STORE MODULE
# ============================================

class VectorStore:
    """Manage embeddings and vector database"""
    
    def __init__(self):
        print("\n🧠 Initializing embedding models...")
        
        # Load embedding model (multilingual)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"  ✓ Loaded: {config.EMBEDDING_MODEL}")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        
        try:
            self.collection = self.client.get_collection("ensakh_docs")
            print(f"  ✓ Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name="ensakh_docs",
                metadata={"hnsw:space": "cosine"}
            )
            print("  ✓ Created new collection")
        
        # BM25 for keyword search
        self.bm25 = None
        self.documents = []
    
    def index_documents(self, documents: List[Dict]):
        """Index documents in vector store"""
        print("\n🔍 Indexing documents...")
        
        texts = [doc['text'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        # Add to ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        # Initialize BM25
        tokenized_docs = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        
        print(f"✅ Indexed {len(documents)} chunks")
    
    def search(self, query: str, k: int = config.TOP_K) -> List[Dict]:
        """Hybrid search combining semantic + keyword"""
        # Normalize query
        processor = TextProcessor()
        normalized_query = processor.normalize_query(query)
        
        # 1. Semantic search (ChromaDB)
        query_embedding = self.embedder.encode([normalized_query])[0].tolist()
        
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2
        )
        
        # 2. Keyword search (BM25)
        tokenized_query = normalized_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k * 2]
        
        # 3. Merge results
        results = []
        seen_ids = set()
        
        # Add semantic results
        for i, doc_id in enumerate(semantic_results['ids'][0]):
            if doc_id not in seen_ids:
                results.append({
                    'id': doc_id,
                    'text': semantic_results['documents'][0][i],
                    'metadata': semantic_results['metadatas'][0][i],
                    'score': semantic_results['distances'][0][i],
                    'method': 'semantic'
                })
                seen_ids.add(doc_id)
        
        # Add keyword results
        for idx in top_bm25_indices:
            doc = self.documents[idx]
            if doc['id'] not in seen_ids and bm25_scores[idx] > 0:
                results.append({
                    'id': doc['id'],
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'score': bm25_scores[idx],
                    'method': 'keyword'
                })
                seen_ids.add(doc['id'])
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]


# ============================================
# 4. RAG INTEGRATION MODULE
# ============================================

class ENSAKHRag:
    """Main RAG system integrating all components"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.processor = TextProcessor()
        
    def build_knowledge_base(self, force_rescrape: bool = False):
        """Build or update knowledge base"""
        print("=" * 60)
        print("🚀 ENSAKH RAG System - Knowledge Base Builder")
        print("=" * 60)
        
        # Check if we have cached data
        cache_file = config.DATA_DIR / "documents.json"
        
        if cache_file.exists() and not force_rescrape:
            print("\n📦 Loading cached documents...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        else:
            # Scrape website
            scraper = ENSAKHScraper()
            documents = scraper.scrape_all()
            
            # Cache documents
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
        
        # Process documents
        processed_docs = self.processor.process_documents(documents)
        
        # Index in vector store
        self.vector_store.index_documents(processed_docs)
        
        print("\n" + "=" * 60)
        print("✅ Knowledge base ready!")
        print("=" * 60)
    
    def retrieve(self, query: str) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context for query"""
        results = self.vector_store.search(query)
        
        # Format context for LLM
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'unknown')
            url = result['metadata'].get('url', '')
            context_parts.append(
                f"[{i}] Source: {source}\n"
                f"URL: {url}\n"
                f"{result['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        return context, results
    
    def generate_prompt(self, query: str, context: str) -> str:
        """Generate prompt for LLaMA model"""
        prompt = f"""Tu es un assistant expert pour l'ENSAKH (École Nationale des Sciences Appliquées de Khouribga).

RÈGLES CRITIQUES:
1. Réponds UNIQUEMENT en utilisant le CONTEXTE fourni ci-dessous
2. Si la réponse N'EST PAS dans le contexte, dis: "Je n'ai pas cette information dans ma base de connaissances"
3. NE JAMAIS inventer ou halluciner des informations
4. TOUJOURS citer les sources avec [1], [2], etc.
5. Supporte les questions en Français, Anglais et Darija

CONTEXTE:
{context}

QUESTION: {query}

Réponds dans la même langue que la question. Sois concis et précis.

RÉPONSE:"""
        
        return prompt


# ============================================
# 5. CLI & TESTING
# ============================================

def test_rag_system():
    """Test the RAG system with sample queries"""
    rag = ENSAKHRag()
    
    test_queries = [
        "what is IID?",
        "who is hafidi?",
        "how can i enter to ensakh?",
        "chkon maleh yassine?",
        "give me the program of API",
        "what departments does ensakh have?",
        "génie informatique c'est quoi?"
    ]
    
    print("\n" + "=" * 60)
    print("🧪 Testing RAG System")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        context, results = rag.retrieve(query)
        
        print(f"\n📊 Retrieved {len(results)} documents:")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result['score']:.3f} | Method: {result['method']}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Text preview: {result['text'][:200]}...")
        
        print(f"\n📝 Generated Prompt:")
        prompt = rag.generate_prompt(query, context)
        print(prompt[:500] + "...")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ENSAKH RAG System")
    parser.add_argument(
        '--build',
        action='store_true',
        help='Build knowledge base'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test RAG system'
    )
    parser.add_argument(
        '--rescrape',
        action='store_true',
        help='Force rescrape website'
    )
    
    args = parser.parse_args()
    
    rag = ENSAKHRag()
    
    if args.build or args.rescrape:
        rag.build_knowledge_base(force_rescrape=args.rescrape)
    
    if args.test:
        test_rag_system()
    
    if not any([args.build, args.test, args.rescrape]):
        parser.print_help()


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES:
===============

1. Build knowledge base (first time):
   python ensakh_rag.py --build

2. Test the system:
   python ensakh_rag.py --test

3. Rebuild with fresh scrape:
   python ensakh_rag.py --build --rescrape

4. Use in your code:
   from ensakh_rag import ENSAKHRag
   
   rag = ENSAKHRag()
   rag.build_knowledge_base()
   
   query = "what is IID?"
   context, results = rag.retrieve(query)
   prompt = rag.generate_prompt(query, context)
   
   # Pass prompt to your LLaMA model
   response = your_llama_model.generate(prompt)
   print(response)
"""
