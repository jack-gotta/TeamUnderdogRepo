"""Baseline FastAPI app for local development and observability notebooks."""

from fastapi import FastAPI
from pydantic import BaseModel


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    echo: str


app = FastAPI(
    title="Mini Wikipedia RAG API",
    version="0.1.0",
    description="Baseline API endpoints for service health and metadata.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Mini Wikipedia RAG API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": app.version}


@app.post("/echo", response_model=EchoResponse)
def echo(body: EchoRequest) -> EchoResponse:
    return EchoResponse(echo=body.message)
