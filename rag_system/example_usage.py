"""
Simple Example Usage of ENSAKH RAG System
Copy and modify this for your own use cases
"""

# ============================================================================
# EXAMPLE 1: Basic Query
# ============================================================================

def example_basic_query():
    """Most basic usage - just ask a question"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    # Initialize (do this once)
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    # Query (do this many times)
    result = rag.query("What is Génie Informatique?")
    print(result['answer'])


# ============================================================================
# EXAMPLE 2: Multiple Questions
# ============================================================================

def example_multiple_questions():
    """Ask multiple questions efficiently"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    # Initialize once
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    # Ask multiple questions
    questions = [
        "What programs does ENSAKH offer?",
        "How do I apply?",
        "What are the admission requirements?"
    ]
    
    for q in questions:
        result = rag.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}\n")
        print("-" * 70)


# ============================================================================
# EXAMPLE 3: With Context (See Retrieved Documents)
# ============================================================================

def example_with_context():
    """See which documents were used to answer"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    result = rag.query(
        question="What is Génie Informatique?",
        return_context=True  # ← Get context
    )
    
    print("Answer:", result['answer'])
    print("\nSources used:")
    for i, chunk in enumerate(result['context'], 1):
        print(f"{i}. {chunk['metadata']['title']} (relevance: {chunk['relevance']:.2%})")


# ============================================================================
# EXAMPLE 4: Custom Parameters
# ============================================================================

def example_custom_parameters():
    """Fine-tune the generation"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    result = rag.query(
        question="Explain machine learning",
        n_context_chunks=5,      # More context
        max_new_tokens=1000,     # Longer answer
        temperature=0.5          # More focused
    )
    
    print(result['answer'])


# ============================================================================
# EXAMPLE 5: REST API Usage (Python)
# ============================================================================

def example_rest_api():
    """Use the REST API instead of Python API"""
    
    import requests
    
    # Make sure API server is running: python api_server.py
    
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "question": "What is ENSAKH?",
            "n_context_chunks": 3,
            "temperature": 0.7,
            "return_context": True
        }
    )
    
    result = response.json()
    print("Answer:", result['answer'])
    
    if 'context' in result:
        print("\nSources:")
        for chunk in result['context']:
            print(f"- {chunk['metadata']['title']}")


# ============================================================================
# EXAMPLE 6: Multilingual Queries
# ============================================================================

def example_multilingual():
    """Ask questions in different languages"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    questions = {
        "English": "What is Génie Informatique?",
        "French": "Qu'est-ce que le Génie Informatique?",
        "Arabic": "ما هو الجيني انفورماتيك؟"
    }
    
    for lang, question in questions.items():
        result = rag.query(question)
        print(f"\n[{lang}]")
        print(f"Q: {question}")
        print(f"A: {result['answer']}")


# ============================================================================
# EXAMPLE 7: Batch Processing
# ============================================================================

def example_batch_processing():
    """Process many questions efficiently"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    import json
    
    # Initialize
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    # Load questions from file
    with open('questions.txt', 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Process all
    results = []
    for i, question in enumerate(questions, 1):
        print(f"Processing {i}/{len(questions)}...")
        result = rag.query(question)
        results.append({
            'question': question,
            'answer': result['answer']
        })
    
    # Save results
    with open('answers.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(results)} questions!")


# ============================================================================
# EXAMPLE 8: Error Handling
# ============================================================================

def example_error_handling():
    """Handle errors gracefully"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    try:
        vector_store = VectorStore(collection_name="ensakh_knowledge")
        rag = ENSAKHRAGEngine(
            model_name="elabbassimouad/LLAMA-ENSAKH",
            vector_store=vector_store
        )
        
        result = rag.query("What is ENSAKH?")
        print(result['answer'])
        
    except FileNotFoundError:
        print("Error: Knowledge base not found. Run: python build_knowledge_base.py")
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 9: Custom Chatbot
# ============================================================================

def example_chatbot():
    """Simple chatbot interface"""
    
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    print("ENSAKH Chatbot (type 'quit' to exit)")
    print("-" * 50)
    
    # Initialize
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    while True:
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            result = rag.query(question)
            print(f"\nBot: {result['answer']}")
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# EXAMPLE 10: Integration with Web App
# ============================================================================

def example_web_integration():
    """Example Flask integration"""
    
    from flask import Flask, request, jsonify
    from rag_engine import ENSAKHRAGEngine
    from vector_store import VectorStore
    
    app = Flask(__name__)
    
    # Initialize RAG (once at startup)
    vector_store = VectorStore(collection_name="ensakh_knowledge")
    rag = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store
    )
    
    @app.route('/ask', methods=['POST'])
    def ask():
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        try:
            result = rag.query(question)
            return jsonify({
                'question': question,
                'answer': result['answer']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    app.run(port=5000)


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Basic Query', example_basic_query),
        '2': ('Multiple Questions', example_multiple_questions),
        '3': ('With Context', example_with_context),
        '4': ('Custom Parameters', example_custom_parameters),
        '5': ('REST API', example_rest_api),
        '6': ('Multilingual', example_multilingual),
        '7': ('Batch Processing', example_batch_processing),
        '8': ('Error Handling', example_error_handling),
        '9': ('Chatbot', example_chatbot),
        '10': ('Web Integration', example_web_integration),
    }
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\n{'='*70}")
            print(f"Running Example {example_num}: {name}")
            print('='*70 + '\n')
            func()
        else:
            print(f"Unknown example: {example_num}")
    else:
        print("\nAvailable Examples:")
        print("="*70)
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\nUsage: python example_usage.py <number>")
        print("Example: python example_usage.py 1")
