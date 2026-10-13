"""
PromptShield — FastAPI Application Factory
==========================================

Creates and configures the FastAPI application instance,
registers routers, middleware, and the full service lifecycle.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.core.config import settings
from backend.core.logging import setup_logging
from backend.api.v1.router import api_router

log = logging.getLogger("promptshield.main")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    setup_logging()
    log.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # ── Startup ───────────────────────────────────────────────────────────

    # 1. Rule-based engine (stateless, instantiated once)
    from backend.services.rule_engine import RuleBasedEngine  # noqa: PLC0415
    rule_engine = RuleBasedEngine()
    app.state.rule_engine = rule_engine
    log.info("Rule-based engine initialised.")

    # 2. Semantic Analyzer (v2 Contextual)
    from backend.services.semantic_analyzer import SemanticAnalyzerService  # noqa: PLC0415
    semantic_analyzer = SemanticAnalyzerService.get_instance()
    app.state.semantic_analyzer = semantic_analyzer

    # 3. ML Classifier (loads model from disk)
    from backend.services.ml_classifier import MLClassifierService  # noqa: PLC0415
    ml_classifier = MLClassifierService.get_instance(
        model_path=settings.FINE_TUNED_MODEL_PATH,
        threshold=settings.ML_THRESHOLD,
    )
    await ml_classifier.load()
    app.state.ml_classifier = ml_classifier

    # 4. Detection service (orchestrates all 3 layers)
    from backend.services.detection_service import DetectionService  # noqa: PLC0415
    detection_service = DetectionService(
        rule_engine=rule_engine,
        semantic_analyzer=semantic_analyzer,
        ml_classifier=ml_classifier,
    )
    app.state.detection_service = detection_service

    log.info("Semantic Analyzer: PromptShield Contextual v2")
    log.info(f"ML Model: {ml_classifier.model_path}")
    log.info("Contextual Detection: ENABLED")

    # 4. Analytics store
    from backend.api.v1.endpoints.analytics import _AnalyticsStore  # noqa: PLC0415
    app.state.analytics_store = _AnalyticsStore()
    log.info("Analytics store initialised.")

    # 5. Secure LLM Gateway
    from backend.services.llm_gateway import SecureLLMGateway  # noqa: PLC0415
    app.state.llm_gateway = SecureLLMGateway(detection_service=detection_service)
    log.info("Secure LLM Gateway initialised.")

    # 5. Database (optional — only if DB_URL configured)
    try:
        from backend.core.database import create_tables  # noqa: PLC0415
        await create_tables()
        log.info("Database tables verified.")
    except Exception as exc:
        log.warning(f"Database initialisation skipped: {exc}")

    log.info("PromptShield startup complete ✓")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("PromptShield shutting down ...")
    MLClassifierService.reset_instance()
    log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "**PromptShield** — Real-Time Multi-Layer Prompt Injection Detection "
            "and Neutralization Framework for Production LLM Applications.\n\n"
            "Layers: Rule-Based Engine → Semantic Analyzer (Phase 8 Contextual) → ML Classifier (DistilBERT fine-tuned) → Decision Engine → Secure LLM Gateway"
        ),
        version=settings.APP_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "PromptShield",
            "url": "https://github.com/promptshield",
        },
        license_info={"name": "MIT"},
    )

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Root redirect ─────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/api/docs",
            "health": "/api/v1/health",
        }

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
