"""
Conftest for ML tests only.

This conftest is intentionally minimal — it does NOT import any backend
modules (FastAPI, SQLAlchemy, structlog, etc.) so that ML tests can run
without the full backend dependency stack being installed.
"""
# No fixtures needed at this level — all ML test fixtures are defined
# inline in each test module using tmp_path (built-in pytest fixture).
