"""
analysis/multiturn_report.py
------------------------------
Report generation for multi-turn contextual jailbreak results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


def save_multiturn_report(results: list, model: str, output_dir: str = "reports") -> Path:
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"multiturn_{ts}.json"

    payload = {
        "meta": {
            "tool":      "SentinelLLM",
            "module":    "multiturn_contextual_jailbreaks",
            "model":     model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total":     len(results),
            "succeeded": sum(1 for r in results if r.success),
        },
        "results": [r.to_dict() for r in results],
    }

    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Multi-turn report saved: {path}")
    return path
