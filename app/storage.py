"""Interaction & feedback storage for Valluvan (backend-selecting facade).

The Streamlit UI logs every answered question and its 👍/👎 feedback through this
module, staying agnostic to *where* the data lands. Two backends implement the
same interface (`log_interaction` / `log_feedback` / `load_interactions`):

  - Postgres (app/db.py) — used for monitoring (Phase 9); works with a local
    docker Postgres or a managed cloud one (Supabase / Neon) via env vars.
  - JSONL (this file) — a zero-dependency fallback to data/interactions.jsonl so
    the app always runs even with no database.

Backend selection (env `MONITORING_DB`, default `auto`):
  - `postgres` : force Postgres (error if unreachable).
  - `jsonl`    : force the local JSONL file.
  - `auto`     : use Postgres if `POSTGRES_HOST` is set and reachable, otherwise
                 fall back to JSONL (with a warning).
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "interactions.jsonl"
_LOCK = threading.Lock()

# Telemetry fields copied from rag.answer()'s result dict.
_TELEMETRY_KEYS = (
    "retrieval_mode",
    "rewritten_query",
    "prompt_variant",
    "model",
    "provider",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- JSONL backend ---------------------------------------------------------
def _jsonl_log_interaction(question: str, result: dict) -> str:
    interaction_id = str(uuid.uuid4())
    record = {
        "id": interaction_id,
        "ts": _utc_now(),
        "question": question,
        "answer": result.get("answer"),
        "kural_nos": [k["kural_no"] for k in result.get("kurals", [])],
        "feedback": None,
    }
    for key in _TELEMETRY_KEYS:
        record[key] = result.get(key)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK, LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return interaction_id


def _jsonl_log_feedback(interaction_id: str, rating: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "id": interaction_id,
        "ts": _utc_now(),
        "feedback": int(rating),
        "event": "feedback",
    }
    with _LOCK, LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _jsonl_load_interactions() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    interactions: dict[str, dict] = {}
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "feedback":
                if row["id"] in interactions:
                    interactions[row["id"]]["feedback"] = row["feedback"]
            else:
                interactions[row["id"]] = row
    return list(interactions.values())


# --- Backend selection -----------------------------------------------------
@lru_cache(maxsize=1)
def _backend():
    """Resolve the storage backend once. Returns the app.db module or None."""
    choice = os.getenv("MONITORING_DB", "auto").lower()
    if choice == "jsonl":
        return None
    want_pg = choice == "postgres" or (
        choice == "auto" and bool(os.getenv("POSTGRES_HOST"))
    )
    if not want_pg:
        return None
    try:
        from app import db

        db.init()
        return db
    except Exception as e:  # noqa: BLE001 - fall back unless Postgres is forced
        if choice == "postgres":
            raise
        print(f"[storage] Postgres unavailable ({e}); using JSONL fallback.")
        return None


def backend_name() -> str:
    return "postgres" if _backend() is not None else "jsonl"


# --- Public interface ------------------------------------------------------
def log_interaction(question: str, result: dict) -> str:
    db = _backend()
    if db is not None:
        return db.log_interaction(question, result)
    return _jsonl_log_interaction(question, result)


def log_feedback(interaction_id: str, rating: int) -> None:
    db = _backend()
    if db is not None:
        return db.log_feedback(interaction_id, rating)
    return _jsonl_log_feedback(interaction_id, rating)


def load_interactions() -> list[dict]:
    db = _backend()
    if db is not None:
        return db.load_interactions()
    return _jsonl_load_interactions()
