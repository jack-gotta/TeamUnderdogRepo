"""
FastAPI server for the Wikipedia RAG system.

This server provides REST API endpoints for the RAG functionality,
including health checks, data ingestion, and question answering.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our model access layer
try:
    from llamaindex_models import get_gpt4o, get_text_embedding_3_large, get_available_models
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model imports failed: {e}")
    MODELS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia RAG System API",
    description="A Retrieval-Augmented Generation system using LlamaIndex with Wikipedia articles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Response models
class HealthResponse(BaseModel):
    status: str
    message: str
    models_available: bool
    timestamp: str

class EchoRequest(BaseModel):
    message: str
    metadata: Optional[Dict[str, Any]] = None

class EchoResponse(BaseModel):
    echo: str
    received_metadata: Optional[Dict[str, Any]] = None
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the service is running.

    Returns:
        HealthResponse: Status information including model availability
    """
    from datetime import datetime

    # Test model availability
    models_working = False
    if MODELS_AVAILABLE:
        try:
            models = get_available_models()
            models_working = len(models) > 0
        except Exception:
            models_working = False

    return HealthResponse(
        status="healthy" if models_working else "degraded",
        message="Service is running" if models_working else "Service running but models unavailable",
        models_available=models_working,
        timestamp=datetime.utcnow().isoformat()
    )

# Echo endpoint for testing
@app.post("/echo", response_model=EchoResponse)
async def echo_message(request: EchoRequest):
    """
    Simple echo endpoint for testing request/response functionality.

    Args:
        request: EchoRequest with message and optional metadata

    Returns:
        EchoResponse: Echoed message with timestamp
    """
    from datetime import datetime

    return EchoResponse(
        echo=request.message,
        received_metadata=request.metadata,
        timestamp=datetime.utcnow().isoformat()
    )

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "Wikipedia RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Model status endpoint
@app.get("/models/status")
async def models_status():
    """
    Get status of available AI models.

    Returns:
        Dict: Model availability and configuration
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    try:
        available_models = get_available_models()
        return {
            "available": True,
            "models": available_models,
            "endpoint": os.getenv("AILAB_ENDPOINT", "https://ct-enterprisechat-api.azure-api.net/")
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model access failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run with: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)