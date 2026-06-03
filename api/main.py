"""
api/main.py
-----------
FastAPI backend for SentinelLLM.
Wraps CLI commands as REST endpoints so the platform can be used
programmatically or integrated into CI/CD pipelines.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import attacks, reports, health

app = FastAPI(
    title="SentinelLLM API",
    description="LLM Security Testing Platform — REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,  prefix="/health",  tags=["Health"])
app.include_router(attacks.router, prefix="/attacks", tags=["Attacks"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])
