"""
Vector Store Module for ENSAKH RAG System
Creates embeddings and stores them in ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages embeddings and vector database
    - Creates embeddings using sentence-transformers
    - Stores vectors in ChromaDB
    - Provides similarity search
    """
    
    def __init__(
        self, 
        collection_name: str = "ensakh_knowledge",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_directory: str = "./chroma_db"
    ):
        """
        Args:
            collection_name: Name for ChromaDB collection
            embedding_model: HuggingFace model for embeddings (multilingual for French/Arabic/English)
            persist_directory: Directory to persist ChromaDB
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("✓ Embedding model loaded")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "ENSAKH knowledge base for RAG system"}
            )
            logger.info(f"✓ Created new collection: {collection_name}")
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Create embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
        
        Returns:
            List of embedding vectors
        """
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info("✓ Embeddings created")
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add documents to vector store
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            batch_size: Batch size for adding to ChromaDB
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract texts and metadata
            texts = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts, batch_size=32)
            
            # Generate IDs
            ids = [f"chunk_{chunk['metadata']['doc_id']}_{chunk['metadata']['chunk_id']}" 
                   for chunk in batch]
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✓ Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        logger.info(f"✅ Successfully added {len(chunks)} chunks to vector store")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Dict = None
    ) -> Dict:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            Dictionary with results
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': self.persist_directory
        }
    
    def delete_collection(self):
        """Delete the collection (use with caution!)"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")


def main():
    """Test the vector store"""
    
    # Load processed chunks
    input_path = Path("processed_chunks.json")
    if not input_path.exists():
        logger.error("Please run document_processor.py first to generate processed_chunks.json")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="ensakh_knowledge",
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_directory="./chroma_db"
    )
    
    # Add documents
    vector_store.add_documents(chunks)
    
    # Show stats
    stats = vector_store.get_collection_stats()
    logger.info(f"\n📊 Collection Stats:")
    logger.info(f"   Name: {stats['collection_name']}")
    logger.info(f"   Total chunks: {stats['total_chunks']}")
    logger.info(f"   Location: {stats['persist_directory']}")
    
    # Test search
    logger.info("\n🔍 Testing search...")
    test_queries = [
        "What is Génie Informatique?",
        "Comment s'inscrire à ENSAKH?",
        "شروط القبول في المدرسة"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = vector_store.search(query, n_results=3)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'], 
            results['metadatas'], 
            results['distances']
        )):
            logger.info(f"\n  Result {i+1} (distance: {distance:.4f}):")
            logger.info(f"  Title: {metadata.get('title', 'N/A')}")
            logger.info(f"  Text: {doc[:150]}...")


if __name__ == "__main__":
    main()
