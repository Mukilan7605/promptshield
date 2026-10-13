# PromptShield

PromptShield is a multi-layer prompt injection detection system built for production LLM applications. It intercepts user prompts before they reach a language model, analyses them across several detection layers, and either flags or blocks attempts to manipulate model behaviour through crafted inputs.

Prompt injection is a class of attack where a user embeds instructions inside their input to override the system's intended behaviour — for example, instructing the model to ignore its guidelines, leak sensitive data, or act outside its defined role. PromptShield addresses this by running each prompt through a detection pipeline before it is forwarded to any downstream model.


## How it works

A prompt submitted to the system passes through three layers in sequence:

**Rule-based engine** — Fast pattern matching using curated regex rules and keyword blocklists. Catches known injection patterns with sub-millisecond latency. No model inference required.

**ML classifier** — A fine-tuned DistilBERT transformer trained on labelled prompt injection datasets. Produces a confidence score indicating the probability that the input is malicious. Handles novel phrasings that rule patterns may miss.

**Semantic analyser** — Analyses the contextual intent of the prompt, looking for indirect or obfuscated injection attempts that evade both rule and token-level detection.

A decision engine combines the outputs of all active layers and returns a verdict: clean, suspicious, or blocked. Every scan is persisted to the database for audit and analytics.


## Stack

**Backend** — Python, FastAPI, SQLAlchemy (async), Alembic, Structlog

**ML** — HuggingFace Transformers, DistilBERT, trained on the deepset prompt injection dataset

**Frontend** — React 18, TypeScript, Vite, Recharts, TanStack Query

**Database** — SQLite for development, PostgreSQL for production (same codebase, switch via environment variable)


## Project structure

```
promptshield/
├── backend/
│   ├── api/v1/endpoints/       # health, detection, analytics, chat
│   ├── core/                   # config, database, logging
│   ├── models/                 # ORM models (scan records)
│   └── services/               # rule engine, ML classifier, semantic analyser, decision engine
├── frontend/
│   └── src/
│       ├── pages/              # Dashboard, Analyze, History
│       ├── components/         # Layout, Sidebar
│       ├── api/                # Typed API client
│       └── types/              # Shared TypeScript types
├── ml/
│   ├── preprocessing.py        # Dataset loading and tokenisation
│   ├── train.py                # Fine-tuning script
│   ├── evaluate.py             # Metrics computation
│   └── inference.py            # Production inference wrapper
├── Datasets/raw/               # Parquet datasets (train, validation, test)
├── migrations/                 # Alembic migration scripts
├── tests/                      # pytest test suite
├── scripts/                    # Evaluation and live-test scripts
├── reports/                    # Per-phase evaluation reports
├── .env.example
├── requirements.txt
└── docker-compose.yml
```


## Setup

Requirements: Python 3.11+, Node.js 20+

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Copy and configure environment variables
copy .env.example .env

# Run database migrations
alembic upgrade head

# Start the backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Start the frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Backend API: http://localhost:8000/api/docs

Frontend dashboard: http://localhost:3000


## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Service health and status |
| POST | /api/v1/detect | Submit a prompt for injection analysis |
| GET | /api/v1/analytics | Aggregated detection statistics |

The detection endpoint returns the verdict, confidence score, which layer triggered the decision, and a sanitised version of the input if applicable.


## Running tests

```bash
pytest
```


## Database

Development uses SQLite with no configuration required. For production, update `DATABASE_URL` in `.env` to a PostgreSQL connection string. No code changes are needed — both use the same SQLAlchemy async interface.


## License

MIT
