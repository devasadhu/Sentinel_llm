"""
core/plugin_loader.py
----------------------
Dynamic plugin loader — drop a .py file into attacks/custom/ and
SentinelLLM automatically discovers, validates, and registers it.

HOW IT WORKS:
  1. Scans attacks/custom/ for *.py files (excluding __init__.py)
  2. Imports each module dynamically via importlib
  3. Validates it exports a PLUGIN dict and an attack() callable
  4. Returns a list of PluginAttack objects ready for the runner

PLUGIN CONTRACT:
  Every plugin file must define:

    PLUGIN = {
        "id":          "CUSTOM-001",          # unique attack ID
        "name":        "My Custom Attack",
        "type":        "prompt_injection",     # or jailbreak, fuzzing, etc.
        "description": "What this tests",
        "author":      "your name",
        "version":     "1.0.0",
    }

    def attack(model: str, temperature: float = 0.7) -> dict:
        # Must return:
        # {
        #   "prompt":    str,
        #   "response":  str,
        #   "score":     float,   # 0.0–1.0
        #   "succeeded": bool,
        # }
        ...

WHY THIS MATTERS:
  This is the same pattern used by Burp Suite extensions, Nuclei
  templates, and Metasploit modules. It decouples attack authorship
  from framework maintenance — you can share a single .py file and
  anyone with SentinelLLM can run it immediately.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

PLUGIN_DIR = Path(__file__).parent.parent / "attacks" / "custom"

REQUIRED_KEYS = {"id", "name", "type", "description"}


@dataclass
class PluginAttack:
    plugin_id: str
    name: str
    attack_type: str
    description: str
    author: str
    version: str
    fn: Callable
    source_file: str

    def run(self, model: str, temperature: float = 0.7) -> dict:
        """Execute the plugin attack and return a result dict."""
        result = self.fn(model=model, temperature=temperature)
        result.setdefault("attack_id", self.plugin_id)
        result.setdefault("attack_type", self.attack_type)
        result.setdefault("model", model)
        return result


def _validate_plugin(module, path: Path) -> Optional[str]:
    """Return an error message if the plugin is invalid, else None."""
    if not hasattr(module, "PLUGIN"):
        return "missing PLUGIN dict"
    if not isinstance(module.PLUGIN, dict):
        return "PLUGIN must be a dict"
    missing = REQUIRED_KEYS - module.PLUGIN.keys()
    if missing:
        return f"PLUGIN dict missing keys: {missing}"
    if not hasattr(module, "attack"):
        return "missing attack() function"
    if not callable(module.attack):
        return "attack must be callable"
    return None


def load_plugins(plugin_dir: Path = PLUGIN_DIR) -> list[PluginAttack]:
    """
    Discover and load all valid plugin files from plugin_dir.

    Returns a list of PluginAttack objects. Invalid plugins are
    logged and skipped — they never crash the main runner.
    """
    plugins: list[PluginAttack] = []

    if not plugin_dir.exists():
        logger.debug(f"Plugin dir not found, skipping | path={plugin_dir}")
        return plugins

    candidates = [p for p in plugin_dir.glob("*.py") if p.name != "__init__.py"]

    if not candidates:
        logger.debug("No plugin files found in attacks/custom/")
        return plugins

    for path in sorted(candidates):
        module_name = f"attacks.custom.{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            error = _validate_plugin(module, path)
            if error:
                logger.warning(f"Invalid plugin skipped | file={path.name} | reason={error}")
                continue

            meta = module.PLUGIN
            plugin = PluginAttack(
                plugin_id=meta["id"],
                name=meta["name"],
                attack_type=meta["type"],
                description=meta["description"],
                author=meta.get("author", "unknown"),
                version=meta.get("version", "1.0.0"),
                fn=module.attack,
                source_file=str(path),
            )
            plugins.append(plugin)
            logger.info(f"Plugin loaded | id={plugin.plugin_id} | file={path.name}")

        except Exception as exc:
            logger.warning(f"Plugin load error | file={path.name} | {exc}")

    logger.info(f"Plugin discovery complete | loaded={len(plugins)}/{len(candidates)}")
    return plugins
