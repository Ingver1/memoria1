"""Application configuration models for the Unified Memory System."""

from __future__ import annotations

import json
import logging.config
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, PositiveInt, ValidationError, field_validator
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

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - simple immutability check
        if self.model_config.get("frozen") and name in self.__dict__:
            raise ValidationError("DatabaseConfig is immutable")
        super().__setattr__(name, value)


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

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Generate key if required
        if self.encrypt_at_rest and not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()
        # Run validators manually
        self._validate_token(self.api_token)
        if self.encryption_key:
            self._validate_key(self.encryption_key)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("SecurityConfig is immutable")
        super().__setattr__(name, value)

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
    rebuild_interval_seconds: PositiveInt = 3600

    model_config = {"frozen": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._workers_range(self.max_workers)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("PerformanceConfig is immutable")
        super().__setattr__(name, value)
    
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

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_port(self.port)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("APIConfig is immutable")
        super().__setattr__(name, value)

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

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_prom_port(self.prom_port)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("MonitoringConfig is immutable")
        super().__setattr__(name, value)

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

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        for p in (self.database.db_path, self.database.vec_path, self.database.cache_path):
            p.parent.mkdir(parents=True, exist_ok=True)

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
            database=DatabaseConfig(connection_pool_size=20),
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
            data = obj.model_dump()
            data.pop("api_token", None)
            data.pop("encryption_key", None)
            if isinstance(obj, SecurityConfig):
                data["has_key"] = bool(self.security.encryption_key)
            return data

        return {
            "version": self.version,
            "profile": self.profile,
            "database": scrub(self.database),
            "model": scrub(self.model),
            "security": scrub(self.security),
            "performance": scrub(self.performance),
            "reliability": scrub(self.reliability),
            "api": scrub(self.api),
            "monitoring": scrub(self.monitoring),
        }

    def save_to_file(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "UnifiedSettings":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def configure_logging(settings: UnifiedSettings | None = None) -> None:
    """Configure basic console logging for the application."""

    settings = settings or UnifiedSettings()

    level = getattr(logging, settings.monitoring.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_settings(env: str | None = None) -> UnifiedSettings:
    env = env or os.getenv("AI_ENV", "development")
    if env == "production":
        return UnifiedSettings.for_production()
    if env == "testing":
        return UnifiedSettings.for_testing()
    if env == "development":
        return UnifiedSettings.for_development()
    return UnifiedSettings()
