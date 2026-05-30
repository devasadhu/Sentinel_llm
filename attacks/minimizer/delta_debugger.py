"""
attacks/minimizer/delta_debugger.py
------------------------------------
Attack Minimization via Hierarchical Delta Debugging.

Given a successful adversarial prompt, finds the minimal subsequence
of tokens/sentences that preserves attack success above a score threshold.

Algorithm: Binary-search-based delta debugging (Zeller 1999) adapted for
           natural language prompts with LLM-as-judge scoring.

Why this design:
- Segment-first (sentences), then token-level within survivors: coarse-to-fine
  avoids wasting LLM calls on fine-grained search in irrelevant regions.
- Immutable seed: we never mutate the original payload object, only derive
  candidates — safe for concurrent use later (asyncio milestone).
- ScoreCache: identical prompt strings are memoized. LLM inference is the
  bottleneck; duplicate candidates appear often in binary search.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


@dataclass
class MinimizationResult:
    """Outcome of a single minimization run."""
    attack_id: str
    original_prompt: str
    minimal_prompt: str
    original_score: float
    minimal_score: float
    original_tokens: int
    minimal_tokens: int
    reduction_ratio: float
    queries_used: int
    duration_seconds: float
    segments_removed: list[str]
    strategy: str
    success: bool


class ScoreCache:
    """
    Memoize (prompt) -> score to avoid duplicate LLM calls.

    Why a plain dict and not functools.lru_cache: we need hit/miss stats
    for the report, and lru_cache doesn't expose those cleanly.
    """

    def __init__(self) -> None:
        self._cache: dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[float]:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: float) -> None:
        self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class DeltaDebugger:
    """
    Hierarchical delta debugger for adversarial prompts.

    Usage:
        debugger = DeltaDebugger(scorer_fn, threshold=0.7)
        result = debugger.minimize("PI-006", original_prompt, attack_type)
    """

    def __init__(
        self,
        scorer_fn: Callable[[str], float],
        threshold: float = 0.7,
        max_queries: int = 80,
        min_segment_chars: int = 8,
    ) -> None:
        self._score = scorer_fn
        self.threshold = threshold
        self.max_queries = max_queries
        self.min_segment_chars = min_segment_chars
        self._cache = ScoreCache()
        self._query_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def minimize(
        self,
        attack_id: str,
        prompt: str,
        attack_type: str = "prompt_injection",
    ) -> MinimizationResult:
        """
        Run hierarchical minimization on a prompt.

        Strategy order:
          1. Sentence-level removal  (coarsest — biggest reduction per query)
          2. Word-level removal      (medium)

        We stop as soon as we can't reduce further or hit max_queries.
        """
        t0 = time.time()
        self._query_count = 0

        original_score = self._cached_score(prompt)
        if original_score < self.threshold:
            logger.warning(
                f"Seed prompt scores {original_score:.3f} < threshold {self.threshold}. "
                "Minimization requires a successful seed."
            )

        logger.info(
            f"Minimizer | {attack_id} | seed_score={original_score:.3f} "
            f"| tokens={self._token_count(prompt)}"
        )

        current = prompt
        strategy_used = "none"

        for granularity, split_fn in [
            ("sentence", self._split_sentences),
            ("word",     self._split_words),
        ]:
            reduced, changed = self._ddmin(current, split_fn)
            if changed:
                current = reduced
                strategy_used = granularity
                logger.info(
                    f"  [{granularity}] reduced to {self._token_count(current)} tokens"
                )
            if self._query_count >= self.max_queries:
                break

        minimal_score = self._cached_score(current)
        duration = time.time() - t0
        orig_tokens = self._token_count(prompt)
        min_tokens = self._token_count(current)

        result = MinimizationResult(
            attack_id=attack_id,
            original_prompt=prompt,
            minimal_prompt=current,
            original_score=original_score,
            minimal_score=minimal_score,
            original_tokens=orig_tokens,
            minimal_tokens=min_tokens,
            reduction_ratio=1.0 - (min_tokens / orig_tokens) if orig_tokens else 0.0,
            queries_used=self._query_count,
            duration_seconds=duration,
            segments_removed=self._diff_segments(prompt, current),
            strategy=strategy_used,
            success=(current != prompt and minimal_score >= self.threshold),
        )

        logger.info(
            f"Minimizer done | {attack_id} | "
            f"{orig_tokens} → {min_tokens} tokens "
            f"({result.reduction_ratio:.1%} reduction) | "
            f"queries={self._query_count} | cache_hit_rate={self._cache.hit_rate:.1%}"
        )
        return result

    # ------------------------------------------------------------------
    # Delta Debugging Core
    # ------------------------------------------------------------------

    def _ddmin(
        self,
        prompt: str,
        split_fn: Callable[[str], list[str]],
    ) -> tuple[str, bool]:
        """
        Binary-search removal of segments.

        Returns (minimized_prompt, did_anything_change).

        Why binary search over individual removal:
        - Removing segments one at a time is O(n) queries.
        - ddmin finds the minimal subset in O(n log n) queries.
        """
        segments = [s for s in split_fn(prompt) if len(s) >= self.min_segment_chars]
        if len(segments) <= 1:
            return prompt, False

        changed = False
        n = 2

        while len(segments) >= 2 and self._query_count < self.max_queries:
            subsets = self._partition(segments, n)
            reduction_found = False

            for subset in subsets:
                candidate_segments = [s for s in segments if s not in subset]
                if not candidate_segments:
                    continue
                candidate = self._rejoin(candidate_segments, split_fn)
                score = self._cached_score(candidate)

                if score >= self.threshold:
                    segments = candidate_segments
                    n = max(n - 1, 2)
                    changed = True
                    reduction_found = True
                    logger.debug(
                        f"  Removed {len(subset)} segment(s) | "
                        f"remaining={len(segments)} | score={score:.3f}"
                    )
                    break

            if not reduction_found:
                if n >= len(segments):
                    break
                n = min(n * 2, len(segments))

        return self._rejoin(segments, split_fn), changed

    # ------------------------------------------------------------------
    # Segmentation strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?\n])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _split_words(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _rejoin(segments: list[str], split_fn: Callable) -> str:
        return " ".join(segments)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cached_score(self, prompt: str) -> float:
        key = prompt.strip()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self._query_count += 1
        score = self._score(prompt)
        self._cache.set(key, score)
        return score

    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _partition(items: list, n: int) -> list[list]:
        k = max(1, len(items) // n)
        return [items[i:i + k] for i in range(0, len(items), k)]

    @staticmethod
    def _diff_segments(original: str, minimal: str) -> list[str]:
        orig_parts = set(re.split(r'\s+', original))
        min_parts  = set(re.split(r'\s+', minimal))
        return sorted(orig_parts - min_parts)
