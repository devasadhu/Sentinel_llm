"""
core/benchmarker.py
-------------------
Multi-model benchmarking for SentinelLLM.
Runs the same attack suites against multiple models sequentially
and produces a comparison report.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from core.llm_client import LLMClient
from core.attack_runner import AttackRunner
from core.logger import logger


@dataclass
class ModelResult:
    model_name: str
    injection_succeeded: int = 0
    injection_total: int = 0
    jailbreak_succeeded: int = 0
    jailbreak_total: int = 0
    injection_avg_score: float = 0.0
    jailbreak_avg_score: float = 0.0
    successful_attacks: list = field(default_factory=list)
    all_results: dict = field(default_factory=dict)

    @property
    def injection_rate(self):
        return self.injection_succeeded / self.injection_total if self.injection_total else 0.0

    @property
    def jailbreak_rate(self):
        return self.jailbreak_succeeded / self.jailbreak_total if self.jailbreak_total else 0.0

    @property
    def overall_vulnerability(self):
        total = self.injection_total + self.jailbreak_total
        succeeded = self.injection_succeeded + self.jailbreak_succeeded
        return succeeded / total if total else 0.0


@dataclass
class BenchmarkReport:
    models: list[ModelResult]
    timestamp: str = ""
    attack_suites: list = field(default_factory=list)

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "attack_suites": self.attack_suites,
            "models": [
                {
                    "model": m.model_name,
                    "injection": {
                        "succeeded": m.injection_succeeded,
                        "total": m.injection_total,
                        "success_rate": round(m.injection_rate, 4),
                        "avg_score": round(m.injection_avg_score, 4),
                    },
                    "jailbreak": {
                        "succeeded": m.jailbreak_succeeded,
                        "total": m.jailbreak_total,
                        "success_rate": round(m.jailbreak_rate, 4),
                        "avg_score": round(m.jailbreak_avg_score, 4),
                    },
                    "overall_vulnerability": round(m.overall_vulnerability, 4),
                    "successful_attacks": m.successful_attacks,
                }
                for m in self.models
            ],
        }


def run_benchmark(models: list[str], suites: list[str] = None) -> BenchmarkReport:
    if suites is None:
        suites = ["injection", "jailbreak"]

    results = []

    for model_name in models:
        logger.info(f"Benchmarking model: {model_name}")
        client = LLMClient(model=model_name)

        if not client.health_check():
            logger.warning(f"Model {model_name} not available, skipping")
            continue

        runner = AttackRunner(client=client)
        model_result = ModelResult(model_name=model_name)

        for suite in suites:
            logger.info(f"  Running {suite} against {model_name}")
            suite_result = runner.run_suite(suite)
            model_result.all_results[suite] = suite_result

            if suite == "injection":
                model_result.injection_succeeded = suite_result.successful_attacks
                model_result.injection_total = suite_result.total_attacks
                model_result.injection_avg_score = suite_result.average_score
            elif suite == "jailbreak":
                model_result.jailbreak_succeeded = suite_result.successful_attacks
                model_result.jailbreak_total = suite_result.total_attacks
                model_result.jailbreak_avg_score = suite_result.average_score

            for r in suite_result.results:
                if r.status.value == "success":
                    model_result.successful_attacks.append(r.attack_id)

        results.append(model_result)
        logger.info(f"  Done: injection={model_result.injection_rate:.1%} jailbreak={model_result.jailbreak_rate:.1%}")

    return BenchmarkReport(
        models=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        attack_suites=suites,
    )
