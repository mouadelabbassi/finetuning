"""
FastAPI Server for ENSAKH RAG System
Provides REST API for querying the RAG system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import logging
from rag_engine import ENSAKHRAGEngine
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ENSAKH RAG API",
    description="Retrieval-Augmented Generation API for ENSAKH Assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine: Optional[ENSAKHRAGEngine] = None


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    n_context_chunks: int = 3
    max_new_tokens: int = 512
    temperature: float = 0.7
    return_context: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: Optional[List[Dict]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_store_loaded: bool
    collection_stats: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    
    try:
        logger.info("🚀 Starting ENSAKH RAG API...")
        
        # Initialize vector store
        logger.info("Loading vector store...")
        vector_store = VectorStore(
            collection_name="ensakh_knowledge",
            persist_directory="./chroma_db"
        )
        
        # Initialize RAG engine
        logger.info("Loading RAG engine...")
        rag_engine = ENSAKHRAGEngine(
            model_name="elabbassimouad/LLAMA-ENSAKH",
            vector_store=vector_store,
            load_in_4bit=True
        )
        
        logger.info("✅ RAG API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG engine: {e}")
        raise


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "ENSAKH RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    model_loaded = rag_engine is not None
    vector_store_loaded = rag_engine.vector_store is not None if model_loaded else False
    
    stats = None
    if vector_store_loaded:
        try:
            stats = rag_engine.vector_store.get_collection_stats()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if model_loaded and vector_store_loaded else "unhealthy",
        model_loaded=model_loaded,
        vector_store_loaded=vector_store_loaded,
        collection_stats=stats
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: QueryRequest with question and parameters
    
    Returns:
        QueryResponse with answer and optional context
    """
    
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        # Query the RAG engine
        result = rag_engine.query(
            question=request.question,
            n_context_chunks=request.n_context_chunks,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            return_context=request.return_context
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict)
async def get_stats():
    """Get system statistics"""
    
    if rag_engine is None or rag_engine.vector_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        stats = rag_engine.vector_store.get_collection_stats()
        return {
            "model": rag_engine.model_name,
            "vector_store": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server"""
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )


if __name__ == "__main__":
    main()
