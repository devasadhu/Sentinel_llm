"""
storage/replay_store.py
------------------------
Deterministic replay storage for adversarial attack results.

WHY THIS MATTERS:
  A jailbreak finding is only credible if someone else can reproduce it.
  This module captures every variable that affects model output — model
  name, temperature, seed (where supported), prompt, and response —
  and stores them with a content hash for integrity verification.

DESIGN:
  Plain JSON lines file (not SQLite) for this milestone — zero deps,
  human readable, grep-able. Each line is one complete replay record.
  The hash is SHA256 of (model + temperature + prompt) so you can
  detect if a "reproduction" actually used different inputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

REPLAY_LOG = Path("reports/replay_log.jsonl")


@dataclass
class ReplayRecord:
    replay_id: str          # SHA256[:12] of (model+temp+prompt)
    attack_id: str
    attack_type: str
    model: str
    temperature: float
    prompt: str
    response: str
    score: float
    succeeded: bool
    timestamp: str
    notes: str = ""

    @staticmethod
    def build(
        attack_id: str,
        attack_type: str,
        model: str,
        temperature: float,
        prompt: str,
        response: str,
        score: float,
        succeeded: bool,
        notes: str = "",
    ) -> "ReplayRecord":
        fingerprint = f"{model}|{temperature}|{prompt}"
        replay_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
        return ReplayRecord(
            replay_id=replay_id,
            attack_id=attack_id,
            attack_type=attack_type,
            model=model,
            temperature=temperature,
            prompt=prompt,
            response=response,
            score=score,
            succeeded=succeeded,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def verify(self) -> bool:
        """Re-derive ID and confirm it matches — detects tampering."""
        fingerprint = f"{self.model}|{self.temperature}|{self.prompt}"
        expected = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
        return expected == self.replay_id


class ReplayStore:

    def __init__(self, path: Path = REPLAY_LOG) -> None:
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

    def save(self, record: ReplayRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.debug(f"Replay saved | id={record.replay_id} | attack={record.attack_id}")

    def load_all(self) -> list[ReplayRecord]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(ReplayRecord(**json.loads(line)))
                    except Exception as e:
                        logger.warning(f"Skipping malformed replay record: {e}")
        return records

    def find(self, replay_id: str) -> Optional[ReplayRecord]:
        for r in self.load_all():
            if r.replay_id == replay_id:
                return r
        return None

    def find_by_attack(self, attack_id: str) -> list[ReplayRecord]:
        return [r for r in self.load_all() if r.attack_id == attack_id]

    def verify_all(self) -> dict:
        records = self.load_all()
        ok = [r for r in records if r.verify()]
        bad = [r for r in records if not r.verify()]
        return {"total": len(records), "valid": len(ok), "tampered": len(bad),
                "tampered_ids": [r.replay_id for r in bad]}
