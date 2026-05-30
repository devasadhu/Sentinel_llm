from dataclasses import dataclass, field
from core.scorer import Scorer
from loguru import logger
from core.llm_client import LLMClient, llm_client as default_client
from attacks.base_attack import AttackStatus
from attacks.prompt_injection.direct_injection import PromptInjectionAttack
from attacks.jailbreaks.role_play import JailbreakAttack

ATTACK_REGISTRY = {
    "injection": PromptInjectionAttack,
    "jailbreak": JailbreakAttack,
}

@dataclass
class SuiteResult:
    suite_name: str
    model_name: str
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    error_attacks: int
    inconclusive_attacks: int
    results: list = field(default_factory=list)
    timestamp: str = ""

    @property
    def success_rate(self):
        return round(self.successful_attacks / self.total_attacks, 4) if self.total_attacks else 0.0

    @property
    def average_score(self):
        return round(sum(r.score for r in self.results) / len(self.results), 4) if self.results else 0.0

    @property
    def risk_summary(self):
        counts = {"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0}
        for r in self.results:
            if r.risk_level in counts: counts[r.risk_level] += 1
        return counts

    def to_dict(self):
        return {
            "suite_name": self.suite_name, "model_name": self.model_name,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_attacks, "successful": self.successful_attacks,
                "failed": self.failed_attacks, "errors": self.error_attacks,
                "inconclusive": self.inconclusive_attacks,
                "success_rate": self.success_rate, "average_score": self.average_score,
            },
            "risk_summary": self.risk_summary,
            "results": [r.to_dict() for r in self.results],
        }

class AttackRunner:
    def __init__(self, client=None, system_prompt=None):
        self.client = client or default_client
        self.system_prompt = system_prompt

    def run_suite(self, attack_name: str, category=None, severity_filter=None) -> SuiteResult:
        from datetime import datetime, timezone
        if attack_name not in ATTACK_REGISTRY:
            raise ValueError(f"Unknown attack: '{attack_name}'. Available: {list(ATTACK_REGISTRY.keys())}")
        logger.info("=" * 50)
        logger.info(f"Suite: {attack_name.upper()} | Model: {self.client.model}")
        logger.info("=" * 50)
        attack = ATTACK_REGISTRY[attack_name](system_prompt=self.system_prompt)
        results = attack.run_suite(self.client, category=category, severity_filter=severity_filter)
        scorer = Scorer()
        for r in results:
            attack_score = scorer.score(
            payload_text=r.payload_text,
            response_text=r.llm_response,
            attack_type=r.attack_type,
            payload_indicators=r.indicators_found,
            )
            r.score = attack_score.score
            r.status = attack_score.status
            r.indicators_found = attack_score.indicators_found
        ok   = [r for r in results if r.status == AttackStatus.SUCCESS]
        fail = [r for r in results if r.status == AttackStatus.FAILURE]
        err  = [r for r in results if r.status == AttackStatus.ERROR]
        inc  = [r for r in results if r.status == AttackStatus.INCONCLUSIVE]
        suite = SuiteResult(
            suite_name=attack_name, model_name=self.client.model,
            total_attacks=len(results), successful_attacks=len(ok),
            failed_attacks=len(fail), error_attacks=len(err),
            inconclusive_attacks=len(inc), results=results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"Done: {len(ok)}/{len(results)} succeeded ({suite.success_rate:.1%}) | avg={suite.average_score:.3f}")
        return suite

    def run_all(self):
        return {name: self.run_suite(name) for name in ATTACK_REGISTRY}

    def health_check(self):
        return self.client.health_check()
