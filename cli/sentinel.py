#!/usr/bin/env python3
import sys
from analysis.transferability import build_transferability_matrix, load_latest_benchmark, save_transferability_report
from analysis.defense_advisor import get_recommendations_from_suite
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.pdf_reporter import generate_pdf_report
from rich.markup import escape

import typer
from rich.console import Console
from core.logger import setup_logger
from core.attack_runner import AttackRunner, ATTACK_REGISTRY
from analysis.report_generator import generate_json_report, print_summary

app = typer.Typer(help="SentinelLLM - AI Security Testing Framework")
console = Console()

@app.command()
def health():
    """Check if Ollama is running and model is available."""
    setup_logger()
    runner = AttackRunner()
    if runner.health_check():
        console.print("[green]✓ Ollama is running and model is available[/green]")
    else:
        console.print("[red]✗ Health check failed. Run: ollama serve[/red]")
        raise typer.Exit(1)

@app.command()
def run(
    attack:        str  = typer.Option("injection", "--attack",        "-a", help="injection | jailbreak | all"),
    system_prompt: str  = typer.Option(None,        "--system-prompt", "-s", help="Custom system prompt to attack"),
    report:        bool = typer.Option(True,        "--report/--no-report",  help="Save JSON report"),
    severity:      str  = typer.Option(None,        "--severity",            help="LOW|MEDIUM|HIGH|CRITICAL"),
):
    """Run an attack suite against the local LLM."""
    setup_logger()
    runner = AttackRunner(system_prompt=system_prompt)
    console.print(f"[bold cyan]SentinelLLM[/bold cyan] - attack: [yellow]{attack}[/yellow]")
    if not runner.health_check():
        console.print("[red]Health check failed. Aborting.[/red]")
        raise typer.Exit(1)
    targets = list(ATTACK_REGISTRY.keys()) if attack == "all" else [attack]
    if attack != "all" and attack not in ATTACK_REGISTRY:
        console.print(f"[red]Unknown attack: {attack}. Choose: {list(ATTACK_REGISTRY.keys())} or all[/red]")
        raise typer.Exit(1)
    for name in targets:
        suite = runner.run_suite(name, severity_filter=severity)
        print_summary(suite)
        if report:
            path = generate_json_report(suite)
            console.print(f"[dim]JSON: {path}[/dim]")
            pdf_path = generate_pdf_report(suite.to_dict())
            console.print(f"[dim]PDF:  {pdf_path}[/dim]")

        recs = get_recommendations_from_suite(suite)
        if recs:
            console.print(f"\n[bold yellow]Defense Recommendations ({len(recs)} findings):[/bold yellow]")
            for r in recs:
                console.print(f"  [red]►[/red] [{r.attack_id}] {r.attack_title}")
                console.print(f"    [dim]Category:[/dim] {r.category}")
                console.print(f"    [dim]Fix:[/dim] {r.remediation}")
                console.print(f"    [dim]Code:[/dim]")
                console.print(f"    [green]{escape(r.code_snippet)}[/green]\n")


@app.command()
def benchmark(
    models: str = typer.Option("llama3.2:1b,llama3.1:8b,qwen2.5:3b", help="Comma-separated model names"),
    suites: str = typer.Option("injection,jailbreak", help="Comma-separated attack suites"),
):
    """Run attack suites against multiple models and compare results."""
    from core.benchmarker import run_benchmark
    import json, os
    from datetime import datetime

    model_list = [m.strip() for m in models.split(",")]
    suite_list = [s.strip() for s in suites.split(",")]

    console.print(f"[cyan]Benchmarking {len(model_list)} models across {len(suite_list)} suites...[/cyan]")

    report = run_benchmark(model_list, suite_list)

    # Print comparison table
    from rich.table import Table
    table = Table(title="Model Vulnerability Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Injection Rate", justify="center")
    table.add_column("Jailbreak Rate", justify="center")
    table.add_column("Avg Score", justify="center")
    table.add_column("Overall Vuln %", justify="center")
    table.add_column("Successful Attacks")

    for m in report.models:
        table.add_row(
            m.model_name,
            f"{m.injection_rate:.1%} ({m.injection_succeeded}/{m.injection_total})",
            f"{m.jailbreak_rate:.1%} ({m.jailbreak_succeeded}/{m.jailbreak_total})",
            f"{((m.injection_avg_score + m.jailbreak_avg_score) / 2):.3f}",
            f"{m.overall_vulnerability:.1%}",
            ", ".join(m.successful_attacks) or "none",
        )

    console.print(table)

    # Save report
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"reports/benchmark_{ts}.json"
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    console.print(f"[green]Report saved: {path}[/green]")

@app.command()
def transferability():
    """Analyze cross-model attack transferability from latest benchmark."""
    from rich.table import Table

    try:
        benchmark = load_latest_benchmark()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    report = build_transferability_matrix(benchmark)
    models = report.models

    console.print(f"\n[bold cyan]Cross-Model Attack Transferability Matrix[/bold cyan]")
    console.print(f"[dim]Models: {', '.join(models)}[/dim]\n")

    table = Table(title="Attack Transferability")
    table.add_column("Attack ID", style="cyan")
    table.add_column("Transferability", justify="center")
    table.add_column("Score", justify="center")
    for model in models:
        table.add_column(model.split(":")[0], justify="center")
    table.add_column("Succeeded On")

    for a in report.attacks:
        color = {
            "UNIVERSAL": "green",
            "HIGH":      "yellow",
            "MEDIUM":    "orange1",
            "LOW":       "red",
        }.get(a.transferability_label, "white")

        table.add_row(
            a.attack_id,
            f"[{color}]{a.transferability_label}[/{color}]",
            f"{a.transferability_score:.2f}",
            *["✅" if a.results_by_model.get(m) else "❌" for m in models],
            ", ".join(a.succeeded_on),
        )

    console.print(table)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Universal attacks (all models):  [green]{len(report.universal_attacks)}[/green]")
    console.print(f"  High transferability (2/3):      [yellow]{len(report.high_transfer_attacks)}[/yellow]")
    console.print(f"  Model-specific (1 model only):   [red]{len(report.model_specific_attacks)}[/red]")
    console.print(f"  Most vulnerable model:           [red]{report.most_vulnerable_model}[/red]")
    console.print(f"  Most resistant model:            [green]{report.most_resistant_model}[/green]")

    path = save_transferability_report(report)
    console.print(f"\n[dim]Report saved: {path}[/dim]")


if __name__ == "__main__":
    app()
