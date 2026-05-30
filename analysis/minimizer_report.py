"""
analysis/minimizer_report.py
-----------------------------
Report generation for minimization results.
Follows the same JSON-to-reports/ pattern as report_generator.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from attacks.minimizer.delta_debugger import MinimizationResult
from loguru import logger


def save_minimization_report(
    results: list[MinimizationResult],
    model: str,
    output_dir: str = "reports",
) -> Path:
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"minimization_{ts}.json"

    payload = {
        "meta": {
            "tool": "SentinelLLM",
            "module": "attack_minimizer",
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_attacks": len(results),
            "successful_minimizations": sum(1 for r in results if r.success),
        },
        "results": [
            {
                "attack_id":        r.attack_id,
                "success":          r.success,
                "original_tokens":  r.original_tokens,
                "minimal_tokens":   r.minimal_tokens,
                "reduction_ratio":  round(r.reduction_ratio, 4),
                "original_score":   round(r.original_score, 4),
                "minimal_score":    round(r.minimal_score, 4),
                "queries_used":     r.queries_used,
                "duration_seconds": round(r.duration_seconds, 2),
                "strategy":         r.strategy,
                "segments_removed": r.segments_removed,
                "original_prompt":  r.original_prompt,
                "minimal_prompt":   r.minimal_prompt,
            }
            for r in results
        ],
    }

    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Minimization report saved: {path}")
    return path
