# 🛡️ PromptShield

> **Real-Time Multi-Layer Prompt Injection Detection and Neutralization Framework for Production LLM Applications**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61dafb?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue?logo=typescript)](https://typescriptlang.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Overview

PromptShield is a production-grade framework for detecting and neutralising **prompt injection attacks** in real-time before they reach a downstream LLM. It combines:

| Layer | Mechanism | Latency |
|---|---|---|
| **Rule-Based Engine** | Regex patterns + keyword blocklists | < 1 ms |
| **ML Classifier** | Fine-tuned DistilBERT transformer | ~20 ms |
| **LLM Guard** *(future)* | LLM-as-a-judge secondary verification | ~500 ms |

---

## 🗂️ Project Structure

```
promptshield/
├── backend/                    # FastAPI application
│   ├── api/v1/
│   │   ├── endpoints/          # Route handlers (health, detection, analytics)
│   │   └── router.py           # Aggregates all endpoint routers
│   ├── core/
│   │   ├── config.py           # Pydantic settings (reads from .env)
│   │   ├── database.py         # SQLAlchemy async engine + session
│   │   ├── logging.py          # Structlog structured logging
│   │   └── security.py         # JWT + bcrypt utilities
│   ├── models/
│   │   └── scan_record.py      # ORM model for audit/analytics
│   ├── services/
│   │   ├── detection_service.py # Multi-layer detection orchestrator
│   │   ├── rule_engine.py       # Rule-based layer (Phase 2)
│   │   └── ml_classifier.py     # ML layer (Phase 2)
│   └── main.py                 # FastAPI app factory + lifespan
│
├── frontend/                   # React + TypeScript dashboard
│   ├── src/
│   │   ├── api/                # Axios API client + typed call functions
│   │   ├── components/layout/  # Layout, Sidebar
│   │   ├── hooks/              # Custom React hooks (Phase 2)
│   │   ├── pages/              # Dashboard, Analyze, History, 404
│   │   ├── stores/             # Zustand state stores (Phase 2)
│   │   ├── types/              # Shared TypeScript types
│   │   ├── App.tsx             # Router configuration
│   │   ├── index.css           # Global design system (CSS custom properties)
│   │   └── main.tsx            # React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
│
├── ml/                         # ML pipeline
│   ├── preprocessing.py        # Dataset loading + tokenisation (Phase 2)
│   ├── train.py                # Fine-tuning script (Phase 2)
│   ├── evaluate.py             # Metrics computation (Phase 2)
│   └── inference.py            # Production inference wrapper (Phase 2)
│
├── migrations/                 # Alembic async DB migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── notebooks/                  # Jupyter EDA & experimentation (Phase 2)
├── tests/                      # pytest test suite
│   ├── api/
│   │   ├── test_health.py
│   │   └── test_detection.py
│   └── conftest.py             # Shared fixtures, in-memory DB
│
├── Datasets/raw/               # ⚠️ DO NOT MODIFY
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
│
├── data/                       # Runtime SQLite DB (gitignored)
├── logs/                       # Runtime log files (gitignored)
├── models/                     # Model checkpoints (gitignored)
│
├── .env.example                # Environment variable template
├── .gitignore
├── alembic.ini                 # Alembic config
├── docker-compose.yml          # Docker Compose (services uncommented per phase)
├── pyproject.toml              # pytest, ruff, mypy config
├── requirements.txt            # Production Python deps
└── requirements-dev.txt        # Dev-only deps (Jupyter, matplotlib, etc.)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+

### 1. Clone & set up environment

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # (optional, for notebooks)

# Copy environment config
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

### 2. Run database migrations

```bash
alembic upgrade head
```

### 3. Start the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/api/docs

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard at: http://localhost:3000

### 5. Run tests

```bash
pytest
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Service health check |
| `POST` | `/api/v1/detect` | Analyse a prompt for injection |
| `GET` | `/api/v1/analytics` | Aggregated detection statistics |

Full interactive docs: **http://localhost:8000/api/docs**

---

## 🗄️ Database

- **Development**: SQLite (`data/promptshield.db`) — zero configuration
- **Production**: Switch to PostgreSQL by updating `DATABASE_URL` in `.env`

No code changes required — both drivers use SQLAlchemy's async API.

---

## 🔭 Roadmap

| Phase | Focus |
|---|---|
| **Phase 1** ✅ | Project structure, configuration, API scaffold |
| **Phase 2** 🔜 | Data preprocessing, ML fine-tuning, rule engine, full API wiring |
| **Phase 3** 🔜 | Full React dashboard, charts, real-time updates |
| **Phase 4** 🔜 | Docker containerisation, CI/CD, LLM Guard layer |

---

## 📄 License

MIT © PromptShield Contributors
