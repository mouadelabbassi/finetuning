"""
RAG Engine Module for ENSAKH System
Integrates vector store with LLAMA-ENSAKH LLM
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import logging
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ENSAKHRAGEngine:
    """
    Complete RAG system that:
    1. Takes user query
    2. Retrieves relevant context from vector store
    3. Feeds context + query to LLAMA-ENSAKH
    4. Returns enhanced answer
    """
    
    def __init__(
        self,
        model_name: str = "elabbassimouad/LLAMA-ENSAKH",
        vector_store: VectorStore = None,
        device: str = "auto",
        load_in_4bit: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model ID for your fine-tuned LLAMA
            vector_store: Initialized VectorStore instance
            device: Device to load model on
            load_in_4bit: Whether to use 4-bit quantization (saves memory)
        """
        self.model_name = model_name
        self.vector_store = vector_store
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("✓ Tokenizer loaded")
        
        # Load model with optional quantization
        logger.info(f"Loading model from {model_name}...")
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
        
        logger.info("✓ Model loaded")
        self.model.eval()
    
    def retrieve_context(
        self, 
        query: str, 
        n_results: int = 3,
        min_relevance: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve relevant context from vector store
        
        Args:
            query: User query
            n_results: Number of chunks to retrieve
            min_relevance: Minimum relevance score (0-1, lower distance = higher relevance)
        
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vector_store:
            logger.warning("No vector store provided, skipping retrieval")
            return []
        
        results = self.vector_store.search(query, n_results=n_results)
        
        # Filter by relevance
        relevant_chunks = []
        for doc, metadata, distance in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            # Convert distance to relevance score (lower distance = higher relevance)
            relevance = 1 - min(distance, 1.0)
            
            if relevance >= min_relevance:
                relevant_chunks.append({
                    'text': doc,
                    'metadata': metadata,
                    'relevance': relevance
                })
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def format_prompt_with_context(
        self, 
        query: str, 
        context_chunks: List[Dict]
    ) -> str:
        """
        Format prompt with retrieved context for LLM
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
        
        Returns:
            Formatted prompt
        """
        # Build context section
        context_text = ""
        if context_chunks:
            context_text = "Context from ENSAKH documentation:\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk['metadata'].get('title', 'Unknown')
                text = chunk['text'][:500]  # Limit context length
                context_text += f"[Source {i}: {source}]\n{text}\n\n"
        
        # Format with LLAMA 3.1 chat template
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are ENSAKH Assistant, a helpful academic chatbot for ENSAKH (École Nationale des Sciences Appliquées de Khouribga).

You have access to the following context from ENSAKH's official documentation. Use this information to provide accurate, helpful answers.

{context_text}

Instructions:
- Answer based on the provided context when relevant
- If the context doesn't contain the answer, use your general knowledge about ENSAKH
- Always respond in the same language as the question (English, French, or Darija)
- Be concise but informative
- If you're not sure, say so<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate answer using LLAMA-ENSAKH
        
        Args:
            prompt: Formatted prompt with context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated answer
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract answer
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "assistant<|end_header_id|>" in full_response:
            answer = full_response.split("assistant<|end_header_id|>")[-1].strip()
        else:
            answer = full_response
        
        return answer
    
    def query(
        self,
        question: str,
        n_context_chunks: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        return_context: bool = False
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            question: User question
            n_context_chunks: Number of context chunks to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_context: Whether to return retrieved context
        
        Returns:
            Dictionary with answer and optionally context
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Question: {question}")
        logger.info(f"{'='*60}")
        
        # Step 1: Retrieve relevant context
        context_chunks = self.retrieve_context(question, n_results=n_context_chunks)
        
        # Step 2: Format prompt with context
        prompt = self.format_prompt_with_context(question, context_chunks)
        
        # Step 3: Generate answer
        answer = self.generate_answer(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        logger.info(f"\nAnswer: {answer}")
        logger.info(f"{'='*60}\n")
        
        result = {
            'question': question,
            'answer': answer
        }
        
        if return_context:
            result['context'] = context_chunks
        
        return result


def main():
    """Test the RAG engine"""
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        collection_name="ensakh_knowledge",
        persist_directory="./chroma_db"
    )
    
    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    rag_engine = ENSAKHRAGEngine(
        model_name="elabbassimouad/LLAMA-ENSAKH",
        vector_store=vector_store,
        load_in_4bit=True  # Use 4-bit quantization to save memory
    )
    
    # Test queries
    test_questions = [
        "What is Génie Informatique at ENSAKH?",
        "Comment puis-je m'inscrire à ENSAKH?",
        "What are the admission requirements?",
        "Quels sont les départements disponibles?",
        "Tell me about the computer engineering program"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("TESTING RAG SYSTEM")
    logger.info("="*60 + "\n")
    
    for question in test_questions:
        result = rag_engine.query(
            question,
            n_context_chunks=3,
            max_new_tokens=300,
            temperature=0.7,
            return_context=True
        )
        
        # Show retrieved context
        if result.get('context'):
            logger.info("📚 Retrieved Context:")
            for i, chunk in enumerate(result['context'], 1):
                logger.info(f"  {i}. {chunk['metadata'].get('title', 'N/A')} "
                          f"(relevance: {chunk['relevance']:.2f})")
        
        logger.info("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
