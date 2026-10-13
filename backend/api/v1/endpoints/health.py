"""
PromptShield — Health Check Endpoint
======================================

GET /api/v1/health        — Liveness probe (always fast)
GET /api/v1/health/ready  — Readiness probe (checks services)
GET /api/v1/health/info   — Extended system info
"""

from __future__ import annotations

import platform
import sys
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.core.config import settings

router = APIRouter()

_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    services: dict[str, str]
    uptime_seconds: float


class SystemInfoResponse(BaseModel):
    app_name: str
    version: str
    environment: str
    python_version: str
    platform: str
    uptime_seconds: float
    ml_model_loaded: bool
    ml_model_path: str
    rule_engine_patterns: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 OK immediately. Used by load balancers to verify the process is alive.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Checks whether all services are initialised and ready to serve traffic.",
)
async def readiness_check(request: Request) -> ReadinessResponse:
    services: dict[str, str] = {}

    # Rule engine
    rule_engine = getattr(request.app.state, "rule_engine", None)
    services["rule_engine"] = "ready" if rule_engine is not None else "unavailable"

    # ML classifier
    ml = getattr(request.app.state, "ml_classifier", None)
    if ml is None:
        services["ml_classifier"] = "unavailable"
    elif getattr(ml, "is_available", False):
        services["ml_classifier"] = "ready"
    else:
        services["ml_classifier"] = "disabled (model not found)"

    # Detection service
    svc = getattr(request.app.state, "detection_service", None)
    services["detection_service"] = "ready" if svc is not None else "unavailable"

    overall = "ready" if all(
        v in ("ready", "disabled (model not found)")
        for v in services.values()
    ) else "degraded"

    return ReadinessResponse(
        status=overall,
        services=services,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get(
    "/info",
    response_model=SystemInfoResponse,
    summary="Extended system information",
)
async def system_info(request: Request) -> SystemInfoResponse:
    ml = getattr(request.app.state, "ml_classifier", None)
    ml_loaded = getattr(ml, "is_available", False) if ml else False
    ml_path = str(getattr(ml, "model_path", settings.FINE_TUNED_MODEL_PATH))

    rule_engine = getattr(request.app.state, "rule_engine", None)
    n_patterns = len(getattr(rule_engine, "rules", [])) if rule_engine else 0

    return SystemInfoResponse(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        python_version=sys.version,
        platform=platform.platform(),
        uptime_seconds=round(time.time() - _START_TIME, 1),
        ml_model_loaded=ml_loaded,
        ml_model_path=ml_path,
        rule_engine_patterns=n_patterns,
    )
