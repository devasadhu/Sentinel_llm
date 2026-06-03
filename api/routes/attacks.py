"""
api/routes/attacks.py
---------------------
Attack execution endpoints. Each endpoint runs the corresponding
attack module and returns structured results.
All attacks run synchronously in a threadpool via run_in_executor
so they don't block the event loop during Ollama inference.
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Request / Response models ─────────────────────────────────────────────────

class AttackRequest(BaseModel):
    model: str = "llama3.2:1b"

class BenchmarkRequest(BaseModel):
    models: list[str] = ["llama3.2:1b"]
    attacks: list[str] = ["injection", "jailbreak"]
    parallel: bool = False
    workers: int = 2

class FuzzRequest(BaseModel):
    model:       str = "llama3.2:1b"
    attack_id:   str = "PI-006"
    attack_type: str = "prompt_injection"
    generations: int = 2
    variants:    int = 3

class MultiTurnRequest(BaseModel):
    model:      str        = "llama3.2:1b"
    attack_ids: list[str]  = ["MT-001", "MT-002", "MT-003", "MT-004"]

class RagRequest(BaseModel):
    model: str = "llama3.2:1b"

class SupplyChainRequest(BaseModel):
    model: str = "llama3.2:1b"

# ── Helper ────────────────────────────────────────────────────────────────────

async def _run_sync(fn, *args, **kwargs):
    """Run a blocking function in the default threadpool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/run/{attack_type}")
async def run_attack(attack_type: Literal["injection", "jailbreak"], body: AttackRequest):
    """Run prompt injection or jailbreak suite against a model."""
    try:
        from core.llm_client import LLMClient
        from core.attack_runner import AttackRunner
        from core.scorer import Scorer

        client  = LLMClient(model=body.model)
        runner  = AttackRunner(client=client)
        results = await _run_sync(runner.run_suite, attack_type)

        return {
            "model":   body.model,
            "attack":  attack_type,
            "total":   len(results),
            "results": [
                {
                    "id":       r.attack_id,
                    "score":    r.score,
                    "complied": r.complied,
                    "payload":  r.payload[:80],
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark")
async def run_benchmark(body: BenchmarkRequest):
    """Run full benchmark across multiple models and attack suites."""
    try:
        from core.benchmarker import run_benchmark as _bench
        report = await _run_sync(_bench, body.models, body.attacks)

        return {
            "models": [
                {
                    "model":                  m.model_name,
                    "overall_vulnerability":  m.overall_vulnerability,
                    "injection_rate":         m.injection_rate,
                    "jailbreak_rate":         m.jailbreak_rate,
                }
                for m in report.models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag")
async def run_rag(body: RagRequest):
    """Run RAG poisoning attack suite."""
    try:
        from attacks.rag.rag_attacker import run_rag_suite
        from analysis.rag_report import save_rag_report

        report = await _run_sync(run_rag_suite, model=body.model)
        path   = save_rag_report(report)

        return {
            "model":          report.model,
            "total":          report.total_attacks,
            "retrieval_rate": report.retrieval_rate,
            "compliance_rate": report.compliance_rate,
            "report_path":    str(path),
            "results": [
                {
                    "id":               r.attack_id,
                    "variant":          r.attack_variant,
                    "query":            r.query,
                    "poison_retrieved": r.poison_retrieved,
                    "score":            r.score,
                    "complied":         r.complied,
                }
                for r in report.results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/supply-chain")
async def run_supply_chain(body: SupplyChainRequest):
    """Audit model artifacts for supply-chain tampering."""
    try:
        from attacks.supply_chain.supply_chain_auditor import run_supply_chain_audit
        from analysis.supply_chain_report import save_supply_chain_report

        report = await _run_sync(run_supply_chain_audit, model=body.model)
        path   = save_supply_chain_report(report)

        return {
            "model":      report.model,
            "risk_level": report.risk_level,
            "total":      len(report.results),
            "passed":     len(report.passed),
            "failed":     len(report.failed),
            "critical":   report.critical_count,
            "report_path": str(path),
            "results": [
                {
                    "id":       r.check_id,
                    "check":    r.check_name,
                    "severity": r.severity.value,
                    "passed":   r.passed,
                    "detail":   r.detail,
                }
                for r in report.results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fuzz")
async def run_fuzz(body: FuzzRequest):
    """Run AutoFuzzer evolutionary attack generation."""
    try:
        from attacks.fuzzer.autofuzzer import AutoFuzzer
        from core.llm_client import LLMClient

        client = LLMClient(model=body.model)
        fuzzer = AutoFuzzer(client=client)
        results = await _run_sync(
            fuzzer.run,
            attack_id=body.attack_id,
            attack_type=body.attack_type,
            generations=body.generations,
            variants=body.variants,
        )
        return {"model": body.model, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multiturn")
async def run_multiturn(body: MultiTurnRequest):
    """Run multi-turn contextual jailbreak attacks."""
    try:
        from attacks.contextual.multiturn_attacker import MultiTurnAttacker
        from core.llm_client import LLMClient

        client   = LLMClient(model=body.model)
        attacker = MultiTurnAttacker(client=client)
        results  = await _run_sync(attacker.run, attack_ids=body.attack_ids)
        return {"model": body.model, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
