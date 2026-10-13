"""
pytest configuration and shared fixtures for PromptShield test suite.

Backend fixtures (async_client, db_session) are registered lazily so that
ML-only test runs don't fail when the full backend stack isn't installed.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backend fixtures — only registered when the full stack is available
# ---------------------------------------------------------------------------

try:
    from typing import AsyncGenerator

    import pytest_asyncio
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from backend.core.database import Base, get_db
    from backend.main import app

    TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

    test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    TestingSessionLocal = async_sessionmaker(
        bind=test_engine, expire_on_commit=False, autoflush=False
    )

    @pytest_asyncio.fixture(scope="session")
    async def init_test_db():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @pytest_asyncio.fixture
    async def db_session(init_test_db) -> AsyncGenerator[AsyncSession, None]:
        async with TestingSessionLocal() as session:
            yield session

    @pytest_asyncio.fixture
    async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
        """HTTP test client with DB dependency overridden to use in-memory DB."""

        async def _override_get_db():
            yield db_session

        app.dependency_overrides[get_db] = _override_get_db
        if not getattr(app.state, "detection_service", None):
            from backend.services.detection_service import DetectionService
            app.state.detection_service = DetectionService()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
        app.dependency_overrides.clear()

except Exception:  # noqa: BLE001
    # Backend stack not available (e.g. running ML-only tests).
    # Backend fixtures simply won't be registered.
    pass
