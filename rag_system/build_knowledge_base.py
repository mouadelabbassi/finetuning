"""
Complete Pipeline to Build ENSAKH Knowledge Base
Runs: Scraping → Processing → Embedding → Storage
"""

import logging
from pathlib import Path
import json
from web_scraper import ENSAKHWebScraper
from document_processor import DocumentProcessor
from vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_knowledge_base(
    start_urls: list,
    max_pages: int = 100,
    max_depth: int = 2,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    collection_name: str = "ensakh_knowledge",
    persist_directory: str = "./chroma_db"
):
    """
    Complete pipeline to build knowledge base
    
    Args:
        start_urls: List of URLs to start crawling
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        collection_name: ChromaDB collection name
        persist_directory: Directory to persist ChromaDB
    """
    
    logger.info("="*80)
    logger.info("ENSAKH RAG KNOWLEDGE BASE BUILDER")
    logger.info("="*80)
    
    # Step 1: Web Scraping
    logger.info("\n📡 STEP 1: WEB SCRAPING")
    logger.info("-"*80)
    
    scraper = ENSAKHWebScraper(max_depth=max_depth)
    documents = scraper.crawl(start_urls, max_pages=max_pages)
    
    # Save scraped documents
    scraped_path = Path("scraped_documents.json")
    with open(scraped_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved {len(documents)} documents to {scraped_path}")
    
    # Step 2: Document Processing
    logger.info("\n🔧 STEP 2: DOCUMENT PROCESSING")
    logger.info("-"*80)
    
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_documents(documents)
    
    # Save processed chunks
    chunks_path = Path("processed_chunks.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved {len(chunks)} chunks to {chunks_path}")
    
    # Step 3: Create Embeddings & Store in ChromaDB
    logger.info("\n🧠 STEP 3: CREATING EMBEDDINGS & STORING IN VECTOR DB")
    logger.info("-"*80)
    
    vector_store = VectorStore(
        collection_name=collection_name,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_directory=persist_directory
    )
    
    vector_store.add_documents(chunks)
    
    # Show final stats
    stats = vector_store.get_collection_stats()
    
    logger.info("\n" + "="*80)
    logger.info("✅ KNOWLEDGE BASE BUILT SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"   • Documents scraped: {len(documents)}")
    logger.info(f"   • Chunks created: {len(chunks)}")
    logger.info(f"   • Vectors stored: {stats['total_chunks']}")
    logger.info(f"   • Collection: {stats['collection_name']}")
    logger.info(f"   • Location: {stats['persist_directory']}")
    logger.info("\n💡 Next step: Run rag_engine.py to test the RAG system!")
    logger.info("="*80 + "\n")


def main():
    """Main execution"""
    
    # ENSAKH URLs to crawl
    start_urls = [
        "http://ensak.usms.ac.ma/ensak/",
        "http://ensak.usms.ac.ma/ensak/formation-initiale/",
        "http://ensak.usms.ac.ma/ensak/formation-continue/",
        "http://ensak.usms.ac.ma/ensak/emplois-du-temps/",
        "http://ensak.usms.ac.ma/ensak/formations-certifiantes/",
        "http://ensak.usms.ac.ma/ensak/departements/",
    ]
    
    # Build knowledge base
    build_knowledge_base(
        start_urls=start_urls,
        max_pages=50,  # Adjust based on your needs
        max_depth=2,
        chunk_size=512,
        chunk_overlap=50,
        collection_name="ensakh_knowledge",
        persist_directory="./chroma_db"
    )


if __name__ == "__main__":
    main()
