"""
attacks/supply_chain/supply_chain_auditor.py
---------------------------------------------
LLM Supply-Chain Security Auditor for SentinelLLM.

WHAT THIS TESTS:
    Supply-chain attacks target the model artifacts themselves rather than
    the model's runtime behaviour. A compromised model blob, tampered
    template, or backdoored tokenizer config can silently alter model
    behaviour before a single prompt is sent.

THREE CHECK CATEGORIES:
    1. IntegrityCheck  — SHA256 of each blob file vs. filename digest.
                         Mismatch = file was modified after download.
    2. TemplateCheck   — Scan the chat template blob for hidden instructions,
                         system prompt overrides, or suspicious patterns.
    3. ConfigCheck     — Parse the model config blob for anomalous fields,
                         unexpected stop tokens, or suspicious metadata.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

BLOBS_DIR = Path("/usr/share/ollama/.ollama/models/blobs")
MANIFESTS_DIR = Path("/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library")

# ── Severity ──────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_id:    str
    check_name:  str
    severity:    Severity
    passed:      bool
    detail:      str
    blob:        Optional[str] = None

@dataclass
class AuditReport:
    model:        str
    results:      list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> list[CheckResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def critical_count(self) -> int:
        return sum(1 for r in self.failed if r.severity == Severity.CRITICAL)

    @property
    def risk_level(self) -> str:
        if self.critical_count > 0:
            return "CRITICAL"
        highs = sum(1 for r in self.failed if r.severity == Severity.HIGH)
        if highs > 0:
            return "HIGH"
        meds = sum(1 for r in self.failed if r.severity == Severity.MEDIUM)
        if meds > 0:
            return "MEDIUM"
        if self.failed:
            return "LOW"
        return "CLEAN"

# ── Suspicious patterns for template scanning ─────────────────────────────────

TEMPLATE_SUSPICIOUS_PATTERNS = [
    (r"ignore (all |previous |prior )?(instructions?|prompts?|rules?)", Severity.CRITICAL,
     "Hidden instruction override in template"),
    (r"you are now (dan|jailbreak|unrestricted|evil|free)", Severity.CRITICAL,
     "Jailbreak persona injection in template"),
    (r"disregard (your |all )?(safety|guidelines|rules|restrictions)", Severity.CRITICAL,
     "Safety bypass instruction in template"),
    (r"(send|exfiltrate|transmit).{0,40}(password|secret|key|token|credential)", Severity.CRITICAL,
     "Credential exfiltration pattern in template"),
    (r"http[s]?://(?!ollama\.ai|llama\.meta\.com|huggingface\.co)[^\s\"'<>]{8,}", Severity.HIGH,
     "Unexpected external URL in template"),
    (r"<\|?(system|hidden|injected)\|?>", Severity.HIGH,
     "Suspicious hidden system tag in template"),
    (r"(always|never) (reveal|tell|say|mention|disclose)", Severity.MEDIUM,
     "Behavioural override instruction in template"),
    (r"(your|the) (real|true|actual) (purpose|goal|mission|objective) is", Severity.MEDIUM,
     "Hidden purpose redefinition in template"),
]

# ── Config anomaly checks ─────────────────────────────────────────────────────

SUSPICIOUS_STOP_TOKENS = [
    "ignore", "jailbreak", "DAN", "OVERRIDE", "SYSTEM_BYPASS"
]

# ── Core auditor ──────────────────────────────────────────────────────────────

class SupplyChainAuditor:
    """Audits Ollama model artifacts for supply-chain tampering."""

    def __init__(self, model: str) -> None:
        self.model = model
        # llama3.2:1b → ("llama3.2", "1b")
        parts = model.split(":")
        self.model_name = parts[0]
        self.model_tag  = parts[1] if len(parts) > 1 else "latest"

    # ── Manifest loading ──────────────────────────────────────────────────────

    def _load_manifest(self) -> Optional[dict]:
        manifest_path = MANIFESTS_DIR / self.model_name / self.model_tag
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            return None
        try:
            return json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            logger.error(f"Manifest JSON parse error: {e}")
            return None

    def _blob_path(self, digest: str) -> Path:
        """Convert digest like 'sha256:abc...' or 'sha256-abc...' to blob path."""
        name = digest.replace(":", "-")
        return BLOBS_DIR / name

    # ── Check 1: Integrity (SHA256 filename vs file content) ──────────────────

    def _check_integrity(self, digest: str, media_type: str) -> CheckResult:
        blob = self._blob_path(digest)
        expected_hash = digest.split("-")[-1] if "-" in digest else digest.split(":")[-1]
        check_id = f"SC-INT-{expected_hash[:8].upper()}"

        if not blob.exists():
            return CheckResult(
                check_id=check_id,
                check_name="Blob Integrity",
                severity=Severity.CRITICAL,
                passed=False,
                detail=f"Blob file missing: {blob.name}",
                blob=blob.name,
            )

        logger.debug(f"Hashing {blob.name} ({blob.stat().st_size / 1e6:.1f} MB)...")
        sha = hashlib.sha256()
        with open(blob, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        actual_hash = sha.hexdigest()

        passed = actual_hash == expected_hash
        return CheckResult(
            check_id=check_id,
            check_name=f"Blob Integrity [{media_type.split('.')[-1]}]",
            severity=Severity.CRITICAL,
            passed=passed,
            detail=(
                "SHA256 matches filename — blob unmodified"
                if passed else
                f"HASH MISMATCH — expected {expected_hash[:16]}... got {actual_hash[:16]}..."
            ),
            blob=blob.name,
        )

    # ── Check 2: Template scanning ────────────────────────────────────────────

    def _check_template(self, digest: str) -> list[CheckResult]:
        blob = self._blob_path(digest)
        results = []

        if not blob.exists():
            return [CheckResult(
                check_id="SC-TPL-MISSING",
                check_name="Template Scan",
                severity=Severity.HIGH,
                passed=False,
                detail="Template blob missing",
                blob=blob.name,
            )]

        template_text = blob.read_text(errors="replace").lower()

        found_any = False
        for pattern, severity, description in TEMPLATE_SUSPICIOUS_PATTERNS:
            matches = re.findall(pattern, template_text, re.IGNORECASE)
            if matches:
                found_any = True
                results.append(CheckResult(
                    check_id=f"SC-TPL-{severity.value[:3]}-{len(results):02d}",
                    check_name="Template Suspicious Pattern",
                    severity=severity,
                    passed=False,
                    detail=f"{description} | pattern='{pattern}' | matches={matches[:3]}",
                    blob=blob.name,
                ))

        if not found_any:
            results.append(CheckResult(
                check_id="SC-TPL-CLEAN",
                check_name="Template Scan",
                severity=Severity.INFO,
                passed=True,
                detail="No suspicious patterns found in chat template",
                blob=blob.name,
            ))

        return results

    # ── Check 3: Config anomalies ─────────────────────────────────────────────

    def _check_config(self, digest: str) -> list[CheckResult]:
        blob = self._blob_path(digest)
        results = []

        if not blob.exists():
            return [CheckResult(
                check_id="SC-CFG-MISSING",
                check_name="Config Check",
                severity=Severity.HIGH,
                passed=False,
                detail="Config blob missing",
                blob=blob.name,
            )]

        try:
            config = json.loads(blob.read_text(errors="replace"))
        except json.JSONDecodeError:
            results.append(CheckResult(
                check_id="SC-CFG-PARSE",
                check_name="Config Parseable",
                severity=Severity.MEDIUM,
                passed=False,
                detail="Config blob is not valid JSON — may be binary or corrupted",
                blob=blob.name,
            ))
            return results

        # Check stop tokens for suspicious values
        stop_tokens = config.get("stop", [])
        suspicious_stops = [t for t in stop_tokens if any(
            s.lower() in str(t).lower() for s in SUSPICIOUS_STOP_TOKENS
        )]
        if suspicious_stops:
            results.append(CheckResult(
                check_id="SC-CFG-STOP",
                check_name="Stop Token Anomaly",
                severity=Severity.HIGH,
                passed=False,
                detail=f"Suspicious stop tokens: {suspicious_stops}",
                blob=blob.name,
            ))
        else:
            results.append(CheckResult(
                check_id="SC-CFG-STOP",
                check_name="Stop Token Check",
                severity=Severity.INFO,
                passed=True,
                detail=f"Stop tokens clean: {stop_tokens}",
                blob=blob.name,
            ))

        # Check for unexpected top-level keys
        known_keys = {
            "general", "llama", "tokenizer", "stop", "model_type",
            "architecture", "file_type", "model_family", "model_families",
            "parameter_size", "quantization_level", "size_label",
            "model_format", "os", "rootfs", "diff_ids",
            "num_attention_heads", "num_hidden_layers", "hidden_size",
            "vocab_size", "context_length", "embedding_length",
            "feed_forward_length", "rope", "attention"
        }
        config_keys = set(str(k).lower() for k in config.keys())
        unexpected = config_keys - known_keys
        if unexpected:
            results.append(CheckResult(
                check_id="SC-CFG-KEYS",
                check_name="Config Key Anomaly",
                severity=Severity.MEDIUM,
                passed=False,
                detail=f"Unexpected config keys (may be benign): {sorted(unexpected)}",
                blob=blob.name,
            ))
        else:
            results.append(CheckResult(
                check_id="SC-CFG-KEYS",
                check_name="Config Keys Check",
                severity=Severity.INFO,
                passed=True,
                detail="All config keys within expected schema",
                blob=blob.name,
            ))

        return results

    # ── Main audit runner ─────────────────────────────────────────────────────

    def run_audit(self) -> AuditReport:
        report = AuditReport(model=self.model)
        manifest = self._load_manifest()

        if manifest is None:
            report.results.append(CheckResult(
                check_id="SC-MANIFEST",
                check_name="Manifest Load",
                severity=Severity.CRITICAL,
                passed=False,
                detail=f"Cannot load manifest for {self.model}",
            ))
            return report

        layers = manifest.get("layers", [])
        config_digest = manifest.get("config", {}).get("digest", "")

        for layer in layers:
            digest     = layer["digest"]
            media_type = layer["mediaType"]

            # Integrity check every blob
            report.results.append(self._check_integrity(digest, media_type))

            # Template-specific scan
            if "template" in media_type:
                report.results.extend(self._check_template(digest))

        # Config integrity + anomaly checks
        if config_digest:
            report.results.append(self._check_integrity(config_digest, "config"))
            report.results.extend(self._check_config(config_digest))

        return report


def run_supply_chain_audit(model: str) -> AuditReport:
    """Entry point called by CLI."""
    auditor = SupplyChainAuditor(model)
    return auditor.run_audit()
