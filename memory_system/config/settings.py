"""Application configuration models for the Unified Memory System."""

from __future__ import annotations

import json
import logging.config
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings

__all__ = [
    "DatabaseConfig",
    "ModelConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "ReliabilityConfig",
    "APIConfig",
    "MonitoringConfig",
    "UnifiedSettings",
    "configure_logging",
    "get_settings",
]


class DatabaseConfig(BaseModel):
    """Paths and connection limits for the storage backend."""

    db_path: Path = Path("data/memory.db")
    vec_path: Path = Path("data/memory.vectors")
    cache_path: Path = Path("data/memory.cache")
    connection_pool_size: PositiveInt = 10

    model_config = {"frozen": True}


class ModelConfig(BaseModel):
    """Embedding model and ANN index parameters."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_add_size: PositiveInt = 128
    hnsw_m: PositiveInt = 32
    hnsw_ef_construction: PositiveInt = 200
    hnsw_ef_search: PositiveInt = 100
    vector_dim: PositiveInt = 384

    model_config = {"frozen": True}


class SecurityConfig(BaseModel):
    """Security related options."""

    encrypt_at_rest: bool = False
    encryption_key: str = ""
    filter_pii: bool = True
    max_text_length: PositiveInt = 10_000
    rate_limit_per_minute: PositiveInt = 1_000
    api_token: str = "your-secret-token-change-me"

    model_config = {"frozen": True}

    @field_validator("encryption_key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        if not value:
            return value
        try:
            Fernet(value.encode())
        except (ValueError, InvalidToken) as exc:  # pragma: no cover
            raise ValueError("Invalid encryption key") from exc
        return value

    @field_validator("api_token")
    @classmethod
    def _validate_token(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("API token must be at least 8 characters long")
        return value


class PerformanceConfig(BaseModel):
    """Tuning knobs for throughput and caching."""

    max_workers: PositiveInt = 4
    cache_size: PositiveInt = 1_000
    cache_ttl_seconds: PositiveInt = 300
    rebuild_inrs: PositiveInt = 24

    model_config = {"frozen": True}
    
    @field_validator("max_workers")
    @classmethod
    def _workers_range(cls, value: int) -> int:
        if value < 1 or value > 32:
            raise ValueError("max_workers must be between 1 and 32")
        return value


class ReliabilityConfig(BaseModel):
    """Retries and backup settings."""

    max_retries: PositiveInt = 3
    retry_delay_seconds: float = 1.0
    backup_enabled: bool = True
    backup_interval_hours: PositiveInt = 24

    model_config = {"frozen": True}


class APIConfig(BaseModel):
    """HTTP API options."""

    host: str = "0.0.0.0"
    port: PositiveInt = 8000
    enable_cors: bool = True
    enable_api: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = {"frozen": True}

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if value < 0 or value > 65_535 or (value != 0 and value < 1024):
            raise ValueError("port must be between 1024 and 65535")
        return value


class MonitoringConfig(BaseModel):
    """Metrics and diagnostics configuration."""

    enable_metrics: bool = True
    enable_rate_limiting: bool = True
    prom_port: PositiveInt = 9_100
    health_check_interval: PositiveInt = 30
    log_level: str = "INFO"

    model_config = {"frozen": True}

    @field_validator("prom_port")
    @classmethod
    def _validate_prom_port(cls, value: int) -> int:
        if value < 1024 or value > 65_535:
            raise ValueError("prom_port must be between 1024 and 65535")
        return value


class UnifiedSettings(BaseSettings):
    """Aggregate all configuration sections."""

    version: str = "0.8.0a0"
    profile: str = "development"
    database: DatabaseConfig = DatabaseConfig()
    model: ModelConfig = ModelConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    api: APIConfig = APIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    model_config = {"env_prefix": "AI_", "env_file": ".env"}

    @classmethod
    def for_testing(cls) -> "UnifiedSettings":
        return cls(
            profile="testing",
            performance=PerformanceConfig(max_workers=2, cache_size=100, cache_ttl_seconds=10),
            monitoring=MonitoringConfig(enable_metrics=False, health_check_interval=5),
            api=APIConfig(port=0),
            security=SecurityConfig(api_token="test-token-12345678", rate_limit_per_minute=10_000),
        )
        
    @classmethod
    def for_production(cls) -> "UnifiedSettings":
        return cls(
            profile="production",
            performance=PerformanceConfig(max_workers=8, cache_size=5_000),
            security=SecurityConfig(encrypt_at_rest=True, filter_pii=True),
        )
    
    @classmethod
    def for_development(cls) -> "UnifiedSettings":
        return cls(
            profile="development",
            database=DatabaseConfig(connection_pool_size=5),
            performance=PerformanceConfig(max_workers=2, cache_size=500),
            monitoring=MonitoringConfig(enable_metrics=True, log_level="DEBUG", health_check_interval=10),
            api=APIConfig(enable_cors=True),
        )

    def get_database_url(self) -> str:
        return f"sqlite:///{self.database.db_path}"

    def validate_production_ready(self) -> list[str]:
        issues: list[str] = []
        if not self.security.api_token or self.security.api_token == "your-secret-token-change-me":
            issues.append("API token is not set")
        return issues

    def get_config_summary(self) -> dict[str, Any]:
        def scrub(obj: BaseModel) -> dict[str, Any]:
            data: dict[str, Any] = json.loads(obj.model_dump_json())
            data.pop("encryption_key", None)
            data.pop("api_token", None)
            data["has_key"] = bool(getattr(obj, "encryption_key", ""))
            return data

        return {
            "database": scrub(self.database),
            "model": scrub(self.model),
            "security": scrub(self.security),
            "performance": scrub(self.performance),
            "reliability": scrub(self.reliability),
            "api": scrub(self.api),
            "monitoring": scrub(self.monitoring),
        }

    def save_to_file(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, path: Path) -> "UnifiedSettings":
        data = json.loads(path.read_text())
        return cls(**data)


def configure_logging(settings: UnifiedSettings) -> None:
    """Apply ``logging.yaml`` and optionally switch to JSON handlers."""

    from importlib import resources

    import yaml # type: ignore[import]

    cfg_path = resources.files("memory_system") / "config" / "logging.yaml"
    with cfg_path.open("r", encoding="utf-8") as fp:
        logging_cfg = yaml.safe_load(fp)

    if os.getenv("LOG_JSON") == "1":
        logging_cfg["root"]["handlers"] = ["json_console"]

    logging.config.dictConfig(logging_cfg)


_settings: UnifiedSettings | None = None


def get_settings() -> UnifiedSettings:
    """Return a cached ``UnifiedSettings`` instance."""
    global _settings
    if _settings is None:
        _settings = UnifiedSettings()
    return _settings
