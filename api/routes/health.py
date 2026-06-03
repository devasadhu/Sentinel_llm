"""
api/routes/health.py
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    ollama: str
    models: list[str]

@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check SentinelLLM API and Ollama connectivity."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_status = "ok"
    except Exception as e:
        models = []
        ollama_status = f"unreachable: {e}"

    return HealthResponse(
        status="ok",
        ollama=ollama_status,
        models=models,
    )
