"""
ORM model for a prompt scan event.

Stores every prompt that goes through the detection pipeline
along with the verdict and metadata for analytics and auditing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class ScanRecord(Base):
    """Represents a single prompt scan result in the database."""

    __tablename__ = "scan_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True, comment="SHA-256 of the raw prompt")
    prompt_preview: Mapped[str] = mapped_column(String(200), comment="First 200 chars for display")

    # Detection verdict
    is_injection: Mapped[bool] = mapped_column(Boolean, default=False)
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), default="low")  # low|medium|high|critical

    # Per-layer results stored as JSON string (portable across DBs)
    layers_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata
    model_version: Mapped[str] = mapped_column(String(32), default="v1")
    client_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
