"""Health-check and monitoring endpoints."""

from __future__ import annotations

import logging
import platform
import sys
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from memory_system.api.middleware import check_dependencies, session_tracker
from memory_system.api.schemas import HealthResponse, StatsResponse
from memory_system.config.settings import UnifiedSettings, get_settings
from memory_system.core.store import EnhancedMemoryStore
from memory_system.utils.metrics import get_metrics_content_type, get_prometheus_metrics

log = logging.getLogger(__name__)
router = APIRouter(tags=["Health & Monitoring"])
from starlette.responses import Response

# Dependency helpers for route functions


async def _store() -> EnhancedMemoryStore:
    """Dependency to get the global EnhancedMemoryStore (async)."""
    return EnhancedMemoryStore(UnifiedSettings())
    

def _settings() -> UnifiedSettings:
    """Dependency to get current UnifiedSettings."""
    return get_settings()


# Root endpoint (basic service info)


@router.get("/", summary="Service info")
async def root() -> dict[str, Any]:
    """Root health endpoint providing service information."""
    return {
        "service": "Unified Memory System",
        "version": "0.8.0a0",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "api_version": "v1",
    }


# Full health check and liveness/readiness probes


@router.get("/health", response_model=HealthResponse, summary="Full health check")
async def health_check(
    memory_store: EnhancedMemoryStore,
    settings: UnifiedSettings,
) -> HealthResponse:
    """Perform an in-depth health check of the system (components and dependencies)."""
    try:
        component = await memory_store.get_health()
        deps = await check_dependencies()
        overall_ok = component.healthy and all(deps.values())
        status = "healthy" if overall_ok else "degraded"
        stats = await memory_store.get_stats()
        return HealthResponse(
            status=status,
            timestamp=datetime.now(UTC).isoformat(),
            uptime_seconds=component.uptime,
            version="0.8.0a0",
            checks={**component.checks, **deps},
            memory_store_health={
                "total_memories": stats.get("total_memories", 0),
                "index_size": stats.get("index_size", 0),
                "cache_hit_rate": stats.get("cache_stats", {}).get("hit_rate", 0.0),
                "buffer_size": stats.get("buffer_size", 0),
            },
            api_enabled=settings.api.enable_api,
        )
    except Exception as e:
        log.error("Health check failed: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            uptime_seconds=0,
            version="0.8.0a0",
            checks={"health_check_error": False},
            memory_store_health={"error": str(e)},
            api_enabled=settings.api.enable_api,
        )


@router.get("/health/live", summary="Liveness probe")
async def liveness_probe() -> dict[str, str]:
    """Simple liveness probe endpoint (always returns alive if reachable)."""
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/health/ready", summary="Readiness probe")
async def readiness_probe(
    memory_store: EnhancedMemoryStore,
) -> dict[str, Any]:
    """Readiness probe to check if the memory store is ready for requests."""
    component = await memory_store.get_health()
    if component.healthy:
        return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}
    raise HTTPException(status_code=503, detail=f"Service not ready: {component.message}")


@router.get("/stats", response_model=StatsResponse, summary="System statistics")
async def get_stats(
    memory_store: EnhancedMemoryStore,
    settings: UnifiedSettings,
) -> StatsResponse:
    """Retrieve current system and memory store statistics."""
    stats = await memory_store.get_stats()
    current_time = datetime.now(UTC).timestamp()
    active = sum(1 for ts in session_tracker.values() if ts > current_time - 3600)
    return StatsResponse(
        total_memories=stats.get("total_memories", 0),
        active_sessions=active,
        uptime_seconds=stats.get("uptime_seconds", 0),
        memory_store_stats=stats,
        api_stats={
            "cors_enabled": settings.api.enable_cors,
            "rate_limiting_enabled": settings.monitoring.enable_rate_limiting,
            "metrics_enabled": settings.monitoring.enable_metrics,
            "encryption_enabled": settings.security.encrypt_at_rest,
            "pii_filtering_enabled": settings.security.filter_pii,
            "backup_enabled": settings.reliability.backup_enabled,
            "model_name": settings.model.model_name,
            "api_version": "v1",
        },
    )


@router.get("/metrics", summary="Prometheus metrics")
async def metrics_endpoint(settings: UnifiedSettings) -> Response:
    """Expose Prometheus metrics if enabled, otherwise 404."""
    if not settings.monitoring.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return Response(content=get_prometheus_metrics(), media_type=get_metrics_content_type())


@router.get("/version", summary="Version info")
async def get_version() -> dict[str, Any]:
    """Get version and environment details of the running service."""
    return {
        "version": "0.8.0a0",
        "api_version": "v1",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }
