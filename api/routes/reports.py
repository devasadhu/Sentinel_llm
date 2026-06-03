"""
api/routes/reports.py
---------------------
Report retrieval endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

REPORTS_DIR = Path("reports")


def _list_reports(prefix: str) -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    files = sorted(REPORTS_DIR.glob(f"{prefix}_*.json"), reverse=True)
    return [{"filename": f.name, "path": str(f)} for f in files]


def _load_report(filename: str) -> dict:
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")
    return json.loads(path.read_text())


@router.get("/")
async def list_all_reports():
    """List all available reports grouped by type."""
    return {
        "benchmark":    _list_reports("benchmark"),
        "rag":          _list_reports("rag"),
        "supply_chain": _list_reports("supply_chain"),
        "drift":        _list_reports("drift"),
        "multiturn":    _list_reports("multiturn"),
        "scorecard":    _list_reports("scorecard"),
    }


@router.get("/{filename}")
async def get_report(filename: str):
    """Retrieve a specific report by filename."""
    return _load_report(filename)


@router.get("/latest/{report_type}")
async def get_latest_report(report_type: str):
    """Get the most recent report of a given type."""
    reports = _list_reports(report_type)
    if not reports:
        raise HTTPException(status_code=404, detail=f"No {report_type} reports found")
    return _load_report(reports[0]["filename"])
