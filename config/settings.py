from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:1b")
    groq_api_key: str = ""
    groq_judge_model: str = "llama-3.3-70b-versatile" 
    ollama_timeout: int = Field(default=60)
    database_url: str = Field(default=f"sqlite:///{PROJECT_ROOT}/sentinellm.db")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_file: str = Field(default=str(PROJECT_ROOT / "logs" / "sentinel.log"))
    attack_max_retries: int = Field(default=2)
    attack_temperature: float = Field(default=0.1)
    score_success_threshold: float = Field(default=0.7)
    reports_dir: str = Field(default=str(PROJECT_ROOT / "reports"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
