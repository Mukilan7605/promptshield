"""
PromptShield — Analytics Endpoint
====================================

Provides aggregated detection statistics for the dashboard.

Currently backed by an in-memory store that accumulates during the server
lifetime. In production, this will query the SQLite/PostgreSQL scan_records
table via SQLAlchemy.

GET /api/v1/analytics              — Aggregated summary
GET /api/v1/analytics/risk-levels  — Breakdown by risk level
GET /api/v1/analytics/recent       — Last N detection events
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory analytics store (shared via app.state in production use)
# Replaced with DB queries in Phase 3.
# ---------------------------------------------------------------------------

class _AnalyticsStore:
    """Thread-safe-enough in-memory store for single-worker mode."""

    def __init__(self, max_events: int = 10_000) -> None:
        self.total_scans: int = 0
        self.injections_detected: int = 0
        self.total_confidence: float = 0.0
        self.total_processing_ms: float = 0.0
        self.risk_counts: dict[str, int] = defaultdict(int)
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._start_time: float = time.time()

    def record(self, result: dict[str, Any], prompt_preview: str = "") -> None:
        self.total_scans += 1
        if result["is_injection"]:
            self.injections_detected += 1
        self.total_confidence += result["overall_confidence"]
        self.total_processing_ms += result["processing_time_ms"]
        self.risk_counts[result["risk_level"]] += 1
        self._events.append({
            "timestamp": time.time(),
            "is_injection": result["is_injection"],
            "confidence": result["overall_confidence"],
            "risk_level": result["risk_level"],
            "processing_ms": result["processing_time_ms"],
            "prompt_preview": prompt_preview[:80] if prompt_preview else "",
        })

    @property
    def detection_rate(self) -> float:
        if self.total_scans == 0:
            return 0.0
        return round(self.injections_detected / self.total_scans, 4)

    @property
    def avg_confidence(self) -> float:
        if self.total_scans == 0:
            return 0.0
        return round(self.total_confidence / self.total_scans, 4)

    @property
    def avg_processing_ms(self) -> float:
        if self.total_scans == 0:
            return 0.0
        return round(self.total_processing_ms / self.total_scans, 2)

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 1)

    def recent_events(self, n: int = 20) -> list[dict[str, Any]]:
        events = list(self._events)
        return list(reversed(events))[:n]


# Singleton store — attached to app.state at startup
_store = _AnalyticsStore()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyticsSummary(BaseModel):
    total_scans: int
    injections_detected: int
    benign_count: int
    detection_rate: float
    avg_confidence: float
    avg_processing_time_ms: float
    uptime_seconds: float


class RiskLevelBreakdown(BaseModel):
    low: int
    medium: int
    high: int
    critical: int
    total: int


class RecentEvent(BaseModel):
    timestamp: float
    is_injection: bool
    confidence: float
    risk_level: str
    processing_ms: float
    prompt_preview: str


class RecentEventsResponse(BaseModel):
    events: list[RecentEvent]
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_store(request: Request) -> _AnalyticsStore:
    return getattr(request.app.state, "analytics_store", _store)


import logging
from sqlalchemy import select, func
from backend.core.database import AsyncSessionLocal
from backend.models.scan_record import ScanRecord

log = logging.getLogger("promptshield.api.analytics")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=AnalyticsSummary,
    summary="Detection analytics summary",
    description="Aggregated statistics from persistent database scan records.",
)
async def get_analytics(request: Request) -> AnalyticsSummary:
    store = _get_store(request)
    try:
        async with AsyncSessionLocal() as session:
            q_tot = await session.execute(select(func.count(ScanRecord.id)))
            total_scans = q_tot.scalar() or 0

            if total_scans > 0:
                q_inj = await session.execute(select(func.count(ScanRecord.id)).where(ScanRecord.is_injection == True))
                injections_detected = q_inj.scalar() or 0

                q_conf = await session.execute(select(func.avg(ScanRecord.overall_confidence)))
                avg_confidence = round(float(q_conf.scalar() or 0.0), 4)

                q_lat = await session.execute(select(func.avg(ScanRecord.processing_time_ms)))
                avg_processing_time_ms = round(float(q_lat.scalar() or 0.0), 2)

                benign_count = total_scans - injections_detected
                detection_rate = round(injections_detected / total_scans, 4)

                return AnalyticsSummary(
                    total_scans=total_scans,
                    injections_detected=injections_detected,
                    benign_count=benign_count,
                    detection_rate=detection_rate,
                    avg_confidence=avg_confidence,
                    avg_processing_time_ms=avg_processing_time_ms,
                    uptime_seconds=store.uptime_seconds,
                )
    except Exception as exc:
        log.warning(f"Database analytics query fallback: {exc}")

    return AnalyticsSummary(
        total_scans=store.total_scans,
        injections_detected=store.injections_detected,
        benign_count=store.total_scans - store.injections_detected,
        detection_rate=store.detection_rate,
        avg_confidence=store.avg_confidence,
        avg_processing_time_ms=store.avg_processing_ms,
        uptime_seconds=store.uptime_seconds,
    )


@router.get(
    "/risk-levels",
    response_model=RiskLevelBreakdown,
    summary="Breakdown of detections by risk level",
)
async def get_risk_breakdown(request: Request) -> RiskLevelBreakdown:
    store = _get_store(request)
    try:
        async with AsyncSessionLocal() as session:
            q = await session.execute(
                select(ScanRecord.risk_level, func.count(ScanRecord.id)).group_by(ScanRecord.risk_level)
            )
            counts = dict(q.all())
            if counts:
                low = counts.get("low", 0)
                medium = counts.get("medium", 0)
                high = counts.get("high", 0)
                critical = counts.get("critical", 0)
                total = sum(counts.values())
                return RiskLevelBreakdown(
                    low=low,
                    medium=medium,
                    high=high,
                    critical=critical,
                    total=total,
                )
    except Exception as exc:
        log.warning(f"Database risk breakdown query fallback: {exc}")

    return RiskLevelBreakdown(
        low=store.risk_counts.get("low", 0),
        medium=store.risk_counts.get("medium", 0),
        high=store.risk_counts.get("high", 0),
        critical=store.risk_counts.get("critical", 0),
        total=store.total_scans,
    )


@router.get(
    "/recent",
    response_model=RecentEventsResponse,
    summary="Most recent detection events",
)
async def get_recent_events(
    request: Request,
    n: int = Query(default=20, ge=1, le=100, description="Number of events to return"),
) -> RecentEventsResponse:
    store = _get_store(request)
    try:
        async with AsyncSessionLocal() as session:
            q = await session.execute(
                select(ScanRecord).order_by(ScanRecord.created_at.desc()).limit(n)
            )
            records = q.scalars().all()
            if records:
                events = [
                    RecentEvent(
                        timestamp=r.created_at.timestamp() if r.created_at else time.time(),
                        is_injection=r.is_injection,
                        confidence=r.overall_confidence,
                        risk_level=r.risk_level,
                        processing_ms=r.processing_time_ms,
                        prompt_preview=r.prompt_preview,
                    )
                    for r in records
                ]
                return RecentEventsResponse(events=events, count=len(events))
    except Exception as exc:
        log.warning(f"Database recent events query fallback: {exc}")

    raw = store.recent_events(n)
    return RecentEventsResponse(
        events=[RecentEvent(**e) for e in raw],
        count=len(raw),
    )

