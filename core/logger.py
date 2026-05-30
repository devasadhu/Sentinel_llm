import sys
from pathlib import Path
from loguru import logger
from config.settings import settings

def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
    )
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )
    logger.info("SentinelLLM logger initialized")

def log_attack_start(attack_type: str, payload_preview: str) -> None:
    preview = payload_preview[:80] + "..." if len(payload_preview) > 80 else payload_preview
    logger.info(f"[ATTACK_START] type={attack_type} | payload='{preview}'")

def log_attack_result(attack_type: str, score: float, success: bool) -> None:
    status = "SUCCESS" if success else "FAILURE"
    log_fn = logger.warning if success else logger.info
    log_fn(f"[ATTACK_RESULT] type={attack_type} | score={score:.3f} | status={status}")

def log_llm_response(model: str, response_length: int, latency_ms: float) -> None:
    logger.debug(f"[LLM_RESPONSE] model={model} | chars={response_length} | latency_ms={latency_ms:.1f}")

def log_error(component: str, error: Exception) -> None:
    logger.error(f"[ERROR] component={component} | {type(error).__name__}: {error}")
