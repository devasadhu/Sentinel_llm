"""
analysis/transferability.py
---------------------------
Cross-model attack transferability analysis.

For every attack that succeeded on at least one model, checks whether
it also succeeded on other models. Produces a transferability matrix
showing which attacks are model-agnostic vs model-specific.

This is a research-grade finding — transferable attacks are more dangerous
because they work regardless of which model a target deploys.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class AttackTransferability:
    attack_id: str
    results_by_model: dict      # {model_name: bool}  True = succeeded
    transferability_score: float  # 0.0 to 1.0
    succeeded_on: list
    failed_on: list

    @property
    def transferability_label(self):
        s = self.transferability_score
        if s == 1.0:
            return "UNIVERSAL"    # works on all models
        elif s >= 0.67:
            return "HIGH"
        elif s >= 0.34:
            return "MEDIUM"
        else:
            return "LOW"          # works on only one model

    def to_dict(self):
        return {
            "attack_id":             self.attack_id,
            "results_by_model":      self.results_by_model,
            "transferability_score": round(self.transferability_score, 3),
            "transferability_label": self.transferability_label,
            "succeeded_on":          self.succeeded_on,
            "failed_on":             self.failed_on,
        }


@dataclass
class TransferabilityReport:
    models: list
    attacks: list[AttackTransferability]
    timestamp: str = ""

    @property
    def universal_attacks(self):
        return [a for a in self.attacks if a.transferability_label == "UNIVERSAL"]

    @property
    def high_transfer_attacks(self):
        return [a for a in self.attacks if a.transferability_label == "HIGH"]

    @property
    def model_specific_attacks(self):
        return [a for a in self.attacks if a.transferability_label == "LOW"]

    @property
    def most_vulnerable_model(self):
        counts = {}
        for a in self.attacks:
            for model, succeeded in a.results_by_model.items():
                if succeeded:
                    counts[model] = counts.get(model, 0) + 1
        return max(counts, key=counts.get) if counts else None

    @property
    def most_resistant_model(self):
        counts = {}
        for a in self.attacks:
            for model in self.models:
                succeeded = a.results_by_model.get(model, False)
                if not succeeded:
                    counts[model] = counts.get(model, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def to_dict(self):
        return {
            "timestamp":              self.timestamp,
            "models":                 self.models,
            "summary": {
                "total_attacks_tested":   len(self.attacks),
                "universal_attacks":      len(self.universal_attacks),
                "high_transfer_attacks":  len(self.high_transfer_attacks),
                "model_specific_attacks": len(self.model_specific_attacks),
                "most_vulnerable_model":  self.most_vulnerable_model,
                "most_resistant_model":   self.most_resistant_model,
            },
            "attacks": [a.to_dict() for a in self.attacks],
        }


def build_transferability_matrix(benchmark_report: dict) -> TransferabilityReport:
    """
    Build transferability matrix from a benchmark report dict.
    Only includes attacks that succeeded on at least one model.
    """
    models = [m["model"] for m in benchmark_report["models"]]

    # Build {model: set(successful_attack_ids)}
    model_successes = {}
    for m in benchmark_report["models"]:
        model_successes[m["model"]] = set(m["successful_attacks"])

    # Collect all attacks that succeeded on at least one model
    all_successful = set()
    for successes in model_successes.values():
        all_successful.update(successes)

    attacks = []
    for attack_id in sorted(all_successful):
        results_by_model = {
            model: (attack_id in model_successes[model])
            for model in models
        }
        succeeded_on = [m for m, s in results_by_model.items() if s]
        failed_on    = [m for m, s in results_by_model.items() if not s]
        score = len(succeeded_on) / len(models)

        attacks.append(AttackTransferability(
            attack_id=attack_id,
            results_by_model=results_by_model,
            transferability_score=score,
            succeeded_on=succeeded_on,
            failed_on=failed_on,
        ))

    # Sort by transferability score descending
    attacks.sort(key=lambda a: a.transferability_score, reverse=True)

    return TransferabilityReport(
        models=models,
        attacks=attacks,
        timestamp=benchmark_report.get("timestamp", ""),
    )


def load_latest_benchmark() -> dict:
    reports_dir = Path(__file__).parent.parent / "reports"
    files = sorted(reports_dir.glob("benchmark_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError("No benchmark reports found. Run: python -m cli.sentinel benchmark")
    with open(files[0]) as f:
        return json.load(f)


def save_transferability_report(report: TransferabilityReport) -> str:
    from datetime import datetime
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"transferability_{ts}.json"
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return str(path)
