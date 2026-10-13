"""
SQLAlchemy async engine, session factory, and declarative Base.

Architecture notes
──────────────────
- Uses SQLAlchemy 2.x async API throughout.
- Defaults to SQLite via aiosqlite for development.
- Switching to PostgreSQL (asyncpg) requires only a DATABASE_URL change.
- All models must inherit from `Base`.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import settings


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_connect_args: dict = {}
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite requires check_same_thread=False when used with asyncio
    _connect_args = {"check_same_thread": False}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Declarative Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Dependency — yields an async session per request
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all database tables using declarative metadata."""
    import backend.models.scan_record  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_scan_record(result: dict, prompt: str, client_ip: str | None = None) -> None:
    """Save a prompt scan result directly into the database."""
    import hashlib
    import json
    from datetime import datetime, timezone
    from backend.models.scan_record import ScanRecord

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_preview = prompt[:200]

    is_inj = bool(result.get("is_injection", False))
    conf = float(result.get("overall_confidence", result.get("confidence", 0.0)))
    risk_level = str(result.get("risk_level", "low"))
    layers = result.get("layers", [])

    layers_data = []
    if layers:
        for l in layers:
            if isinstance(l, dict):
                layers_data.append(l)
            elif hasattr(l, "model_dump"):
                layers_data.append(l.model_dump())
            elif hasattr(l, "dict"):
                layers_data.append(l.dict())
    layers_json = json.dumps(layers_data) if layers_data else None

    proc_time = float(result.get("processing_time_ms", 0.0))

    record = ScanRecord(
        prompt_hash=prompt_hash,
        prompt_preview=prompt_preview,
        is_injection=is_inj,
        overall_confidence=conf,
        risk_level=risk_level,
        layers_json=layers_json,
        processing_time_ms=proc_time,
        model_version=str(result.get("model_version", "v1")),
        client_ip=client_ip,
        created_at=datetime.now(timezone.utc),
    )

    try:
        async with AsyncSessionLocal() as session:
            session.add(record)
            await session.commit()
    except Exception as exc:
        import logging
        logging.getLogger("promptshield.database").error(f"Failed to persist scan record: {exc}")

