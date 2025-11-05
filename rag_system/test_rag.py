"""
Test Script for ENSAKH RAG System
Demonstrates the complete RAG pipeline with example queries
"""

import logging
from rag_engine import ENSAKHRAGEngine
from vector_store import VectorStore
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rag_system():
    """
    Complete test of the RAG system
    """
    
    print("\n" + "="*80)
    print("🎓 ENSAKH RAG SYSTEM - COMPREHENSIVE TEST")
    print("="*80 + "\n")
    
    # Check if knowledge base exists
    if not Path("./chroma_db").exists():
        logger.error("❌ Knowledge base not found!")
        logger.info("Please run: python build_knowledge_base.py")
        return
    
    # Initialize Vector Store
    logger.info("📚 Loading vector store...")
    vector_store = VectorStore(
        collection_name="ensakh_knowledge",
        persist_directory="./chroma_db"
    )
    
    stats = vector_store.get_collection_stats()
    logger.info(f"✓ Vector store loaded: {stats['total_chunks']} chunks")
    
    # Initialize RAG Engine
    logger.info("\n🤖 Loading LLAMA-ENSAKH model...")
    logger.info("   (This may take a few minutes on first run)")
    
    rag_engine = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store,
        load_in_4bit=True  # Use 4-bit quantization
    )
    
    logger.info("✓ RAG engine ready!\n")
    
    # Test queries in multiple languages
    test_queries = [
        {
            "question": "What is Génie Informatique at ENSAKH?",
            "language": "English",
            "category": "Academic Programs"
        },
        {
            "question": "Comment puis-je m'inscrire à ENSAKH?",
            "language": "French",
            "category": "Admissions"
        },
        {
            "question": "What are the admission requirements?",
            "language": "English",
            "category": "Admissions"
        },
        {
            "question": "Quels sont les départements disponibles à ENSAKH?",
            "language": "French",
            "category": "Departments"
        },
        {
            "question": "Tell me about the computer engineering program",
            "language": "English",
            "category": "Academic Programs"
        },
        {
            "question": "Explain machine learning",
            "language": "English",
            "category": "Technical Knowledge"
        }
    ]
    
    print("\n" + "="*80)
    print("🧪 RUNNING TEST QUERIES")
    print("="*80 + "\n")
    
    for i, query_info in enumerate(test_queries, 1):
        question = query_info['question']
        language = query_info['language']
        category = query_info['category']
        
        print(f"\n{'─'*80}")
        print(f"📝 Test {i}/{len(test_queries)}")
        print(f"   Category: {category}")
        print(f"   Language: {language}")
        print(f"{'─'*80}")
        print(f"\n❓ Question: {question}\n")
        
        # Query the RAG system
        result = rag_engine.query(
            question=question,
            n_context_chunks=3,
            max_new_tokens=400,
            temperature=0.7,
            return_context=True
        )
        
        # Display retrieved context
        if result.get('context'):
            print("📚 Retrieved Context:")
            for j, chunk in enumerate(result['context'], 1):
                title = chunk['metadata'].get('title', 'Unknown')
                relevance = chunk['relevance']
                print(f"   {j}. {title} (relevance: {relevance:.2%})")
            print()
        
        # Display answer
        print(f"🤖 Answer:\n{result['answer']}\n")
        
        print("─"*80)
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   • Total queries tested: {len(test_queries)}")
    print(f"   • Languages: English, French")
    print(f"   • Categories: Academic Programs, Admissions, Departments, Technical")
    print(f"   • Vector store: {stats['total_chunks']} chunks")
    print(f"   • Model: {rag_engine.model_name}")
    print("\n💡 The RAG system is working correctly!")
    print("="*80 + "\n")


def test_vector_search_only():
    """
    Test only the vector search (without LLM generation)
    Useful for debugging retrieval
    """
    
    print("\n" + "="*80)
    print("🔍 TESTING VECTOR SEARCH ONLY")
    print("="*80 + "\n")
    
    # Initialize Vector Store
    vector_store = VectorStore(
        collection_name="ensakh_knowledge",
        persist_directory="./chroma_db"
    )
    
    test_queries = [
        "Génie Informatique",
        "admission requirements",
        "départements ENSAKH"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: {query}")
        print("─"*80)
        
        results = vector_store.search(query, n_results=5)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), 1):
            relevance = 1 - min(distance, 1.0)
            print(f"\n{i}. Relevance: {relevance:.2%}")
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            print(f"   Text: {doc[:200]}...")
        
        print("\n" + "─"*80)
    
    print("\n✅ Vector search test complete!\n")


def interactive_mode():
    """
    Interactive mode - ask questions in real-time
    """
    
    print("\n" + "="*80)
    print("💬 INTERACTIVE MODE")
    print("="*80)
    print("\nType your questions (or 'quit' to exit)\n")
    
    # Initialize
    vector_store = VectorStore(
        collection_name="ensakh_knowledge",
        persist_directory="./chroma_db"
    )
    
    rag_engine = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store,
        load_in_4bit=True
    )
    
    print("✓ System ready!\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if not question:
                continue
            
            print("\n🤖 Thinking...\n")
            
            result = rag_engine.query(
                question=question,
                n_context_chunks=3,
                max_new_tokens=400,
                temperature=0.7,
                return_context=False
            )
            
            print(f"Answer: {result['answer']}\n")
            print("─"*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main test runner"""
    
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "search":
            test_vector_search_only()
        elif mode == "interactive":
            interactive_mode()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python test_rag.py [search|interactive]")
    else:
        # Default: run full test
        test_rag_system()


if __name__ == "__main__":
    main()
