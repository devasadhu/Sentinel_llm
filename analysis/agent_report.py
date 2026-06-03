"""
analysis/agent_report.py
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from attacks.agent.agent_attacker import AgentReport

REPORTS_DIR = Path("reports")
console = Console()

def save_agent_report(report: AgentReport) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"agent_{ts}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": report.model,
        "summary": {
            "total": report.total,
            "succeeded": report.succeeded_count,
            "success_rate": round(report.success_rate, 4),
        },
        "results": [
            {
                "attack_id":       r.attack_id,
                "category":        r.attack_category,
                "description":     r.description,
                "user_input":      r.user_input,
                "succeeded":       r.succeeded,
                "score":           r.score,
                "evidence":        r.evidence,
                "tool_calls":      [tc.name for tc in r.trace.tool_calls],
                "turns":           r.trace.turns,
                "error":           r.trace.error,
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path

def print_agent_report(report: AgentReport) -> None:
    console.print(f"\n[bold cyan]Agent Attack Results[/bold cyan] | model=[yellow]{report.model}[/yellow]")
    console.print(
        f"  Total: {report.total}  |  "
        f"Succeeded: {report.succeeded_count}/{report.total} "
        f"({report.success_rate*100:.1f}%)\n"
    )
    table = Table(title="Agent Attack Results", box=box.ROUNDED)
    table.add_column("ID",       style="cyan", no_wrap=True)
    table.add_column("Category", style="blue", no_wrap=True)
    table.add_column("Description", max_width=35)
    table.add_column("Tools Called", max_width=30)
    table.add_column("Score", no_wrap=True)
    table.add_column("Result", no_wrap=True)

    for r in report.results:
        tools  = ", ".join(r.trace.tool_calls and [tc.name for tc in r.trace.tool_calls] or ["-"])
        result = "[red]SUCCEEDED[/red]" if r.succeeded else "[green]BLOCKED[/green]"
        table.add_row(
            r.attack_id,
            r.attack_category.replace("_", " "),
            r.description[:35] + ("..." if len(r.description) > 35 else ""),
            tools[:30],
            f"{r.score:.3f}",
            result,
        )
    console.print(table)

    by_category: dict[str, list] = {}
    for r in report.results:
        by_category.setdefault(r.attack_category, []).append(r)
    console.print("\n[bold]By category:[/bold]")
    for cat, results in by_category.items():
        succeeded = sum(1 for r in results if r.succeeded)
        console.print(f"  {cat.replace('_',' '):<25} {succeeded}/{len(results)} succeeded")
