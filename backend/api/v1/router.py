"""
API v1 — root router.

All feature sub-routers are registered here with their URL prefixes.
"""

from fastapi import APIRouter

from backend.api.v1.endpoints import health, detection, analytics, chat

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(detection.router, prefix="/detect", tags=["Detection"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(chat.router, prefix="/chat", tags=["Secure LLM Gateway"])

