"""
Quick Start Script for ENSAKH RAG System
Minimal example to get started quickly
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_demo():
    """
    Quick demonstration of the RAG system
    This is a simplified version for quick testing
    """
    
    print("\n" + "="*70)
    print("🚀 ENSAKH RAG SYSTEM - QUICK START")
    print("="*70 + "\n")
    
    # Step 1: Check if knowledge base exists
    if not Path("./chroma_db").exists():
        logger.info("📚 Knowledge base not found. Building it now...")
        logger.info("   This will take a few minutes...\n")
        
        from build_knowledge_base import build_knowledge_base
        
        start_urls = [
            "http://ensak.usms.ac.ma/ensak/",
            "http://ensak.usms.ac.ma/ensak/formation-initiale/",
            "http://ensak.usms.ac.ma/ensak/formation-continue/",
        ]
        
        build_knowledge_base(
            start_urls=start_urls,
            max_pages=20,  # Start small
            max_depth=1
        )
    else:
        logger.info("✓ Knowledge base found!\n")
    
    # Step 2: Initialize RAG system
    logger.info("🤖 Loading RAG system...")
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    vector_store = VectorStore(
        collection_name="ensakh_knowledge",
        persist_directory="./chroma_db"
    )
    
    rag_engine = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store,
        load_in_4bit=True
    )
    
    logger.info("✓ System ready!\n")
    
    # Step 3: Demo queries
    demo_questions = [
        "What is ENSAKH?",
        "Tell me about Génie Informatique",
        "How do I apply to ENSAKH?"
    ]
    
    print("="*70)
    print("📝 DEMO QUERIES")
    print("="*70 + "\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-"*70)
        
        result = rag_engine.query(
            question=question,
            n_context_chunks=2,
            max_new_tokens=300,
            temperature=0.7
        )
        
        print(f"\nAnswer: {result['answer']}\n")
        print("="*70)
    
    print("\n✅ Demo complete!")
    print("\n💡 Next steps:")
    print("   • Run 'python test_rag.py' for comprehensive tests")
    print("   • Run 'python api_server.py' to start the API")
    print("   • Run 'python test_rag.py interactive' for interactive mode\n")


if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user\n")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Make sure all dependencies are installed:")
        logger.info("   pip install -r requirements.txt")
        logger.info("2. Check your internet connection")
        logger.info("3. Verify HuggingFace access to elabbassimouad/LLAMA-ENSAKH")
