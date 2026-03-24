"""
Sprint 1 FastAPI Application - Basic Infrastructure

Simple FastAPI server with baseline endpoints:
- Health check
- Echo endpoint for testing
- Auto-generated documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import sys
from pathlib import Path

# Request/Response Models
class EchoRequest(BaseModel):
    """Simple request model for echo testing."""
    message: str


class EchoResponse(BaseModel):
    """Simple response model for echo testing."""
    echo: str
    received_at: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    python_version: str
    dependencies_loaded: bool


# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia RAG System - Sprint 1",
    description="Basic FastAPI infrastructure with health check and echo endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic system information."""
    return {
        "message": "Wikipedia RAG System - Sprint 1",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Validates that the server is running and basic dependencies are available.
    """
    try:
        # Check if we can import key dependencies
        dependencies_loaded = True
        try:
            import pandas
            import fastapi
            # Don't import heavy dependencies for basic health check
        except ImportError:
            dependencies_loaded = False

        from datetime import datetime

        return HealthResponse(
            status="healthy",
            message="Server is running successfully",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            dependencies_loaded=dependencies_loaded
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            python_version="unknown",
            dependencies_loaded=False
        )


@app.post("/echo", response_model=EchoResponse)
async def echo_endpoint(request: EchoRequest):
    """
    Simple echo endpoint for testing request/response functionality.

    Takes a message and echoes it back with a timestamp.
    """
    from datetime import datetime

    return EchoResponse(
        echo=request.message,
        received_at=datetime.now().isoformat()
    )


@app.get("/echo/{message}")
async def echo_get(message: str):
    """
    GET version of echo endpoint for simple browser testing.
    """
    from datetime import datetime

    return {
        "echo": message,
        "received_at": datetime.now().isoformat(),
        "method": "GET"
    }


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
