"""
Document Processor Module for ENSAKH RAG System
Cleans, chunks, and deduplicates documents
"""

import re
from typing import List, Dict
import hashlib
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process documents for RAG system:
    - Clean text (remove noise, normalize)
    - Chunk into semantic segments
    - Deduplicate similar chunks
    - Add metadata
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.seen_hashes = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        - Remove excessive whitespace
        - Normalize unicode
        - Remove special characters (keep punctuation)
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def is_low_quality(self, text: str) -> bool:
        """
        Check if text is low quality
        - Too short
        - Too many special characters
        - Repetitive content
        """
        # Too short
        if len(text.split()) < 10:
            return True
        
        # Too many special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\u0600-\u06FF]', text)) / len(text)
        if special_char_ratio > 0.3:
            return True
        
        # Check for repetitive content
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                return True
        
        return False
    
    def semantic_chunking(self, text: str, title: str = "") -> List[str]:
        """
        Split text into semantic chunks
        - Respects sentence boundaries
        - Maintains context with overlap
        - Preserves paragraph structure when possible
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap from end of previous chunk
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-self.chunk_overlap:]) if len(words) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If text is too short, return as single chunk
        if not chunks and text:
            chunks = [text]
        
        # Add title context to first chunk if available
        if title and chunks:
            chunks[0] = f"Title: {title}\n\n{chunks[0]}"
        
        return chunks
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for deduplication"""
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove duplicate or near-duplicate chunks
        Uses hash-based deduplication
        """
        unique_chunks = []
        
        for chunk in chunks:
            chunk_hash = self.get_text_hash(chunk['text'])
            
            if chunk_hash not in self.seen_hashes:
                self.seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
        
        logger.info(f"Deduplication: {len(chunks)} → {len(unique_chunks)} chunks")
        return unique_chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process all documents into chunks with metadata
        
        Args:
            documents: List of documents with 'content', 'title', 'url'
        
        Returns:
            List of processed chunks with metadata
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            title = doc.get('title', '')
            url = doc.get('url', '')
            
            # Clean content
            cleaned_content = self.clean_text(content)
            
            # Skip low quality documents
            if self.is_low_quality(cleaned_content):
                logger.warning(f"Skipping low-quality document: {title[:50]}")
                continue
            
            # Chunk the document
            chunks = self.semantic_chunking(cleaned_content, title)
            
            # Create chunk objects with metadata
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_obj = {
                    'text': chunk_text,
                    'metadata': {
                        'source': url,
                        'title': title,
                        'doc_id': doc_idx,
                        'chunk_id': chunk_idx,
                        'total_chunks': len(chunks),
                        'word_count': len(chunk_text.split())
                    }
                }
                all_chunks.append(chunk_obj)
        
        logger.info(f"✓ Processed {len(documents)} documents into {len(all_chunks)} chunks")
        
        # Deduplicate
        unique_chunks = self.deduplicate_chunks(all_chunks)
        
        return unique_chunks


def main():
    """Test the document processor"""
    import json
    from pathlib import Path
    
    # Load scraped documents
    input_path = Path("scraped_documents.json")
    if not input_path.exists():
        logger.error("Please run web_scraper.py first to generate scraped_documents.json")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Process documents
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    chunks = processor.process_documents(documents)
    
    # Save processed chunks
    output_path = Path("processed_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved {len(chunks)} processed chunks to {output_path}")
    
    # Show sample
    if chunks:
        logger.info("\n📄 Sample chunk:")
        logger.info(f"Text: {chunks[0]['text'][:200]}...")
        logger.info(f"Metadata: {chunks[0]['metadata']}")


if __name__ == "__main__":
    main()
