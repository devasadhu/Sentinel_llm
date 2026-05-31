"""
analysis/safety_layer_report.py
---------------------------------
Report generation for safety layer detection results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


def save_safety_layer_report(profiles: list, output_dir: str = "reports") -> Path:
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"safety_layer_{ts}.json"

    payload = {
        "meta": {
            "tool":      "SentinelLLM",
            "module":    "safety_layer_detection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models":    [p.model for p in profiles],
        },
        "profiles": [p.to_dict() for p in profiles],
    }

    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Safety layer report saved: {path}")
    return path
