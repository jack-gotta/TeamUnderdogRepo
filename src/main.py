from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Mini Wikipedia RAG API", version="0.1.0")


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/echo", response_model=EchoResponse)
def echo(payload: EchoRequest) -> EchoResponse:
    return EchoResponse(message=payload.message)
