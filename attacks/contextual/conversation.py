"""
attacks/contextual/conversation.py
------------------------------------
Conversation state container for multi-turn attacks.

Why a dedicated class instead of a plain list of dicts:
- Encapsulates the Ollama message format so the rest of the code
  never has to know about {"role": ..., "content": ...} structure.
- turn_scores lets us track how the model's compliance drifts across
  turns — that drift curve is the research finding.
- to_ollama() / to_log() give clean separation between the wire format
  and the reporting format.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Turn:
    turn_number: int
    role: str          # "user" | "assistant"
    content: str
    score: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "turn":      self.turn_number,
            "role":      self.role,
            "content":   self.content,
            "score":     round(self.score, 4),
            "timestamp": self.timestamp,
        }


class Conversation:
    """
    Ordered list of turns with helpers for building Ollama message payloads.

    Design note: we store Turn objects (richer) internally but expose
    to_ollama() for the wire format. This lets us add scoring, timestamps,
    and metadata without touching the LLM client code.
    """

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.turns: list[Turn] = []
        self._turn_counter = 0

    def add_user(self, content: str, score: float = 0.0) -> Turn:
        self._turn_counter += 1
        turn = Turn(self._turn_counter, "user", content, score)
        self.turns.append(turn)
        return turn

    def add_assistant(self, content: str, score: float = 0.0) -> Turn:
        self._turn_counter += 1
        turn = Turn(self._turn_counter, "assistant", content, score)
        self.turns.append(turn)
        return turn

    def to_ollama(self) -> list[dict]:
        """Wire format for Ollama /api/chat endpoint."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for turn in self.turns:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def to_log(self) -> list[dict]:
        return [t.to_dict() for t in self.turns]

    @property
    def last_assistant_response(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return ""

    @property
    def turn_count(self) -> int:
        return len(self.turns)
