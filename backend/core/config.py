"""
Application configuration via Pydantic Settings.

All values are read from environment variables (or a .env file).
The settings object is a singleton imported throughout the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for PromptShield."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────
    APP_NAME: str = "PromptShield"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_VERSION: str = "0.1.0"

    # ── API Server ──────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = True
    SECRET_KEY: str = Field(
        default="CHANGE_ME_IN_PRODUCTION_32_CHARS_MIN",
        min_length=32,
    )

    # ── CORS ────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ── Database ────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/promptshield.db"

    # ── ML / Model ──────────────────────────────────────────
    BASE_MODEL_NAME: str = "distilbert-base-uncased"
    MODEL_CACHE_DIR: str = "./models/cache"
    FINE_TUNED_MODEL_PATH: str = "./models/fine_tuned/promptshield_v2/best_model"
    MAX_SEQ_LENGTH: int = 256
    DETECTION_THRESHOLD: float = 0.85
    ML_THRESHOLD: float = 0.50           # Classifier decision threshold (injection class)

    # ── Detection Layers ────────────────────────────────────
    ENABLE_RULE_BASED: bool = True
    ENABLE_SEMANTIC_ANALYSIS: bool = True
    ENABLE_ML_BASED: bool = True
    ENABLE_LLM_GUARD: bool = False

    # ── Decision Engine Weights & Thresholds ────────────────
    WEIGHT_RULE_ENGINE: float = 0.30
    WEIGHT_SEMANTIC_ANALYSIS: float = 0.30
    WEIGHT_ML_CLASSIFIER: float = 0.40

    THRESHOLD_ALLOW_MAX: int = 24       # 0..24  = ALLOW (low)
    THRESHOLD_WARN_MAX: int = 49        # 25..49 = WARN (medium)
    THRESHOLD_SANITIZE_MAX: int = 74    # 50..74 = SANITIZE (high)
    THRESHOLD_BLOCK_MIN: int = 75       # 75..100 = BLOCK (critical)

    ENABLE_SECURITY_OVERRIDE: bool = True

    # ── LLM Gateway Configuration ───────────────────────────
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "mock-model"
    LLM_API_URL: str = ""
    LLM_API_KEY: str = ""

    # ── Rate Limiting ───────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Logging ─────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "./logs/promptshield.log"

    # ── Feature Flags ───────────────────────────────────────
    ENABLE_ANALYTICS_DASHBOARD: bool = True
    ENABLE_AUDIT_LOG: bool = True

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


# Convenience alias used throughout the codebase.
settings = get_settings()
