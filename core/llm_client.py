import time
import requests
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from config.settings import settings
from core.logger import log_llm_response, log_error

@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.text) > 0

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


class LLMClient:
    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        self._generate_url = f"{self.base_url}/api/generate"
        self._tags_url = f"{self.base_url}/api/tags"
        self._version_url = f"{self.base_url}/api/version"
        logger.debug(f"LLMClient initialized | model={self.model}")

    def health_check(self) -> bool:
        try:
            r = requests.get(self._version_url, timeout=5)
            if r.status_code != 200:
                logger.error(f"Ollama returned status {r.status_code}")
                return False
            tags = requests.get(self._tags_url, timeout=5)
            if tags.status_code == 200:
                installed = [m["name"] for m in tags.json().get("models", [])]
                if not any(self.model in m for m in installed):
                    logger.warning(f"Model '{self.model}' not found. Installed: {installed}")
                    return False
            logger.info(f"Health check passed | model={self.model}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Run: ollama serve")
            return False
        except Exception as e:
            log_error("LLMClient.health_check", e)
            return False

    def generate(self, prompt: str, system_prompt=None, temperature=None, max_retries=2) -> LLMResponse:
        temp = temperature if temperature is not None else settings.attack_temperature
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temp, "num_predict": 512}
        }
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                r = requests.post(self._generate_url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                elapsed = (time.time() - t0) * 1000
                data = r.json()
                text = data.get("response", "").strip()
                log_llm_response(self.model, len(text), elapsed)
                return LLMResponse(
                    text=text, model=self.model,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    response_tokens=data.get("eval_count", 0),
                    latency_ms=elapsed, raw_response=data,
                )
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout attempt {attempt+1}/{max_retries+1}")
                if attempt == max_retries:
                    return LLMResponse(text="", model=self.model, error="Timeout")
            except requests.exceptions.ConnectionError:
                return LLMResponse(text="", model=self.model, error="Connection failed")
            except Exception as e:
                log_error("LLMClient.generate", e)
                if attempt == max_retries:
                    return LLMResponse(text="", model=self.model, error=str(e))
            time.sleep(1.0 * (attempt + 1))
        return LLMResponse(text="", model=self.model, error="Unknown error")

    def generate_batch(self, prompts: list, system_prompt=None) -> list:
        return [self.generate(p, system_prompt=system_prompt) for p in prompts]

llm_client = LLMClient()
