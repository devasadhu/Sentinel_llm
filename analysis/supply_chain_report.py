"""
analysis/supply_chain_report.py
"""
from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from attacks.supply_chain.supply_chain_auditor import AuditReport, Severity

REPORTS_DIR = Path("reports")
console = Console()

SEVERITY_COLORS = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH:     "red",
    Severity.MEDIUM:   "yellow",
    Severity.LOW:      "cyan",
    Severity.INFO:     "green",
}

def save_supply_chain_report(report: AuditReport) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"supply_chain_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": report.model,
        "risk_level": report.risk_level,
        "summary": {
            "total_checks": len(report.results),
            "passed": len(report.passed),
            "failed": len(report.failed),
            "critical": report.critical_count,
        },
        "results": [asdict(r) for r in report.results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path

def print_supply_chain_report(report: AuditReport) -> None:
    risk_color = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "cyan",
        "CLEAN": "bold green",
    }.get(report.risk_level, "white")

    console.print(f"\n[bold cyan]Supply Chain Audit[/bold cyan] | model=[yellow]{report.model}[/yellow] | risk=[{risk_color}]{report.risk_level}[/{risk_color}]")
    console.print(f"  Checks: {len(report.results)}  |  Passed: {len(report.passed)}  |  Failed: {len(report.failed)}  |  Critical: {report.critical_count}\n")

    table = Table(title="Supply Chain Audit Results", box=box.ROUNDED)
    table.add_column("ID",         style="cyan",  no_wrap=True)
    table.add_column("Check",      max_width=30)
    table.add_column("Severity",   no_wrap=True)
    table.add_column("Result",     no_wrap=True)
    table.add_column("Detail",     max_width=60)

    for r in report.results:
        color  = SEVERITY_COLORS.get(r.severity, "white")
        result = "[green]PASS[/green]" if r.passed else f"[{color}]FAIL[/{color}]"
        table.add_row(
            r.check_id,
            r.check_name,
            f"[{color}]{r.severity.value}[/{color}]",
            result,
            r.detail[:80] + ("..." if len(r.detail) > 80 else ""),
        )

    console.print(table)
