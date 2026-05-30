from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from loguru import logger

class AttackStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"

@dataclass
class AttackPayload:
    id: str
    name: str
    category: str
    severity: str
    description: str
    payload: str
    expected_behavior: str
    success_indicators: list
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class AttackResult:
    attack_id: str
    attack_name: str
    attack_type: str
    attack_category: str
    payload_text: str
    llm_response: str
    status: AttackStatus
    score: float
    severity: str
    indicators_found: list = field(default_factory=list)
    mitre_tactic_id: str = ""
    mitre_tactic_name: str = ""
    owasp_id: str = ""
    latency_ms: float = 0.0
    timestamp: str = ""
    model_name: str = ""
    error_message: str = ""
    notes: str = ""

    @property
    def is_successful(self):
        return self.status == AttackStatus.SUCCESS

    @property
    def risk_level(self):
        if not self.is_successful:
            return "LOW"
        return {"CRITICAL":"CRITICAL","HIGH":"HIGH","MEDIUM":"MEDIUM","LOW":"LOW"}.get(self.severity,"MEDIUM")

    def to_dict(self):
        return {
            "attack_id": self.attack_id, "attack_name": self.attack_name,
            "attack_type": self.attack_type, "attack_category": self.attack_category,
            "payload_text": self.payload_text, "llm_response": self.llm_response,
            "status": self.status.value, "score": round(self.score, 4),
            "severity": self.severity, "indicators_found": self.indicators_found,
            "mitre_tactic_id": self.mitre_tactic_id, "mitre_tactic_name": self.mitre_tactic_name,
            "owasp_id": self.owasp_id, "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp, "model_name": self.model_name,
            "risk_level": self.risk_level, "error_message": self.error_message,
        }

class BaseAttack(ABC):
    def __init__(self, payloads_path: Path, system_prompt=None):
        self.payloads_path = payloads_path
        self.system_prompt = system_prompt
        self._payloads = []
        self._load_payloads()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def mitre_tactic_id(self) -> str: ...

    @property
    @abstractmethod
    def mitre_tactic_name(self) -> str: ...

    @property
    @abstractmethod
    def owasp_id(self) -> str: ...

    @abstractmethod
    def _evaluate_response(self, payload, response_text): ...

    def _load_payloads(self):
        if not self.payloads_path.exists():
            logger.error(f"Payloads not found: {self.payloads_path}")
            return
        with open(self.payloads_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._payloads = [
            AttackPayload(
                id=p["id"], name=p["name"], category=p["category"],
                severity=p["severity"], description=p["description"],
                payload=p["payload"], expected_behavior=p["expected_behavior"],
                success_indicators=p["success_indicators"], tags=p.get("tags",[]),
            ) for p in raw
        ]
        logger.info(f"Loaded {len(self._payloads)} payloads for {self.name}")

    def get_payloads(self, category=None, severity=None):
        p = self._payloads
        if category: p = [x for x in p if x.category == category]
        if severity: p = [x for x in p if x.severity == severity]
        return p

    def run(self, payload, llm_client):
        from datetime import datetime, timezone
        logger.info(f"Running {payload.id}: {payload.name}")
        response = llm_client.generate(prompt=payload.payload, system_prompt=self.system_prompt)
        if not response.success:
            return AttackResult(
                attack_id=payload.id, attack_name=payload.name, attack_type=self.name,
                attack_category=payload.category, payload_text=payload.payload,
                llm_response="", status=AttackStatus.ERROR, score=0.0,
                severity=payload.severity, error_message=response.error or "Empty response",
                latency_ms=response.latency_ms,
                timestamp=datetime.now(timezone.utc).isoformat(), model_name=response.model,
            )
        status, score, indicators = self._evaluate_response(payload, response.text)
        return AttackResult(
            attack_id=payload.id, attack_name=payload.name, attack_type=self.name,
            attack_category=payload.category, payload_text=payload.payload,
            llm_response=response.text, status=status, score=score,
            severity=payload.severity, indicators_found=indicators,
            mitre_tactic_id=self.mitre_tactic_id, mitre_tactic_name=self.mitre_tactic_name,
            owasp_id=self.owasp_id, latency_ms=response.latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(), model_name=response.model,
        )

    def run_suite(self, llm_client, category=None, severity_filter=None):
        payloads = self.get_payloads(category=category, severity=severity_filter)
        if not payloads:
            logger.warning(f"No payloads for {self.name}")
            return []
        logger.info(f"Running {self.name}: {len(payloads)} payloads")
        results = []
        for p in payloads:
            r = self.run(p, llm_client)
            results.append(r)
            icon = "OK" if r.is_successful else "XX"
            logger.info(f"  [{icon}] {p.id} | score={r.score:.2f} | {r.status.value}")
        ok = sum(1 for r in results if r.is_successful)
        logger.info(f"{self.name} done: {ok}/{len(results)} succeeded")
        return results
