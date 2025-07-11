"""Admin routes exposed under `/api/v1/admin`."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, status
from memory_system.api.middleware import MaintenanceModeMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

router = APIRouter(prefix="/admin", tags=["Administration"])


def _maintenance() -> MaintenanceModeMiddleware:
    """Internal dependency to get the maintenance mode middleware (or raise error)."""
    return MaintenanceModeMiddleware(cast(ASGIApp, None))


@router.get("/maintenance-mode", summary="Get maintenance mode state", response_model=dict)
async def maintenance_status(
    mw: MaintenanceModeMiddleware | None = None,
) -> dict[str, bool]:
    """Check whether maintenance mode is currently enabled."""
    if mw is None:
        mw = MaintenanceModeMiddleware(cast(ASGIApp, None))
    return {"enabled": mw._enabled}


@router.post(
    "/maintenance-mode/enable",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Enable maintenance mode",
)
async def enable_maintenance(
    mw: MaintenanceModeMiddleware | None = None,
) -> Response:
    """Switch maintenance mode **on** (returns 204 No Content on success)."""
    (mw or MaintenanceModeMiddleware(cast(ASGIApp, None))).enable()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/maintenance-mode/disable",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Disable maintenance mode",
)
async def disable_maintenance(
    mw: MaintenanceModeMiddleware | None = None,
) -> Response:
    """Switch maintenance mode **off** and restore normal operation."""
    (mw or MaintenanceModeMiddleware(cast(ASGIApp, None))).disable()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
