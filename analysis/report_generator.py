import json
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
from config.settings import settings

def generate_json_report(suite_result, output_path=None) -> Path:
    reports_dir = Path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = reports_dir / f"report_{suite_result.suite_name}_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(suite_result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"JSON report saved: {output_path}")
    return Path(output_path)

def print_summary(suite_result) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    console.print("\n[bold cyan]SentinelLLM Security Report[/bold cyan]")
    console.print(f"Suite: [yellow]{suite_result.suite_name}[/yellow] | Model: [yellow]{suite_result.model_name}[/yellow]")
    console.print(f"Time:  {suite_result.timestamp}\n")
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("ID",          style="dim",    width=8)
    table.add_column("Attack Name",                 width=30)
    table.add_column("Status",                      width=14)
    table.add_column("Score",       justify="right",width=8)
    table.add_column("Risk",                        width=10)
    table.add_column("MITRE",                       width=12)
    colors = {"success":"green","failure":"red","error":"yellow","partial":"orange3","inconclusive":"blue"}
    for r in suite_result.results:
        c = colors.get(r.status.value, "white")
        risk_str = f"[red]{r.risk_level}[/red]" if r.risk_level in ("CRITICAL","HIGH") else r.risk_level
        table.add_row(r.attack_id, r.attack_name, f"[{c}]{r.status.value.upper()}[/{c}]",
                      f"{r.score:.2f}", risk_str, r.mitre_tactic_id or "N/A")
    console.print(table)
    s = suite_result
    console.print(
        f"\n[bold]Summary:[/bold] {s.total_attacks} attacks | "
        f"[green]{s.successful_attacks} succeeded[/green] | "
        f"[red]{s.failed_attacks} failed[/red] | "
        f"Success rate: [yellow]{s.success_rate:.1%}[/yellow] | "
        f"Avg score: [yellow]{s.average_score:.3f}[/yellow]"
    )
    risk = suite_result.risk_summary
    console.print(f"Risk: CRITICAL={risk['CRITICAL']} HIGH={risk['HIGH']} MEDIUM={risk['MEDIUM']} LOW={risk['LOW']}\n")
