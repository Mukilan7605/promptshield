"""Tests for the health check endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_returns_ok(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_check_version_format(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/health")
    version = response.json()["version"]
    parts = version.split(".")
    assert len(parts) == 3, f"Version should be semver, got: {version}"
