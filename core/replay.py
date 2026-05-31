"""
core/replay.py
---------------
Replay capture hook — wraps attack execution to auto-save replay records.

Usage in attack runner or CLI:
    from core.replay import capture

    with capture(attack_id, attack_type, model, temperature, prompt) as rec:
        response = llm_client.generate(prompt)
        score = scorer.score(...)
        rec.finalize(response.text, score.score, score.score >= 0.7)
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional

from storage.replay_store import ReplayRecord, ReplayStore

_store = ReplayStore()


class _ReplayCapture:
    def __init__(self, attack_id, attack_type, model, temperature, prompt):
        self.attack_id   = attack_id
        self.attack_type = attack_type
        self.model       = model
        self.temperature = temperature
        self.prompt      = prompt
        self._record: Optional[ReplayRecord] = None

    def finalize(self, response: str, score: float, succeeded: bool, notes: str = "") -> ReplayRecord:
        self._record = ReplayRecord.build(
            attack_id=self.attack_id,
            attack_type=self.attack_type,
            model=self.model,
            temperature=self.temperature,
            prompt=self.prompt,
            response=response,
            score=score,
            succeeded=succeeded,
            notes=notes,
        )
        _store.save(self._record)
        return self._record

    @property
    def record(self) -> Optional[ReplayRecord]:
        return self._record


@contextmanager
def capture(attack_id, attack_type, model, temperature, prompt):
    cap = _ReplayCapture(attack_id, attack_type, model, temperature, prompt)
    yield cap
