# Valluvan — Thirukkural Wisdom Assistant
#
# Common commands. Run `make help` to list targets.
# The Python venv is `.venv` (created via uv). PY points at its interpreter.

PY := .venv/bin/python
PIP := uv pip
COMPOSE := docker compose

.DEFAULT_GOAL := help

.PHONY: help venv install env qdrant-up qdrant-down qdrant-logs dashboard \
        normalize ingest search rag eval-ground eval-retrieval eval-rewrite \
        app up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Setup ---
venv: ## Create the uv virtual environment (.venv)
	uv venv .venv

install: ## Install Python dependencies into .venv
	$(PIP) install -r requirements.txt

env: ## Create .env from the template (edit it to add your GROQ_API_KEY)
	cp -n .env.example .env && echo "Created .env — add your GROQ_API_KEY" || echo ".env already exists"

## --- Qdrant (vector DB) ---
qdrant-up: ## Start the Qdrant container
	$(COMPOSE) up -d qdrant

qdrant-down: ## Stop the Qdrant container (keeps data)
	$(COMPOSE) stop qdrant

qdrant-logs: ## Tail Qdrant logs
	$(COMPOSE) logs -f qdrant

dashboard: ## Print the Qdrant web dashboard URL
	@echo "Open http://localhost:6333/dashboard"

## --- Data & ingestion ---
normalize: ## Normalize raw HF parquet -> data/thirukkural.json
	$(PY) ingestion/normalize.py

ingest: ## Embed (dense+sparse) and upsert all 1330 kurals into Qdrant
	$(PY) ingestion/ingest.py

## --- Retrieval / RAG (quick manual checks) ---
search: ## Compare dense/sparse/hybrid retrieval. Usage: make search Q="about anger"
	$(PY) -m rag.search "$(Q)"

rag: ## Ask Valluvan a question. Usage: make rag Q="How can I control my anger?"
	$(PY) -m rag.rag "$(Q)"

## --- Evaluation (Phase 5/6) ---
eval-ground: ## Generate the retrieval ground-truth dataset (LLM)
	$(PY) -m eval.ground_truth

eval-retrieval: ## Evaluate dense/sparse/hybrid/rerank (hit-rate, MRR)
	$(PY) -m eval.eval_retrieval

eval-rewrite: ## Evaluate LLM query rewriting on a sample (raw vs rewritten)
	EVAL_REWRITE=true $(PY) -m eval.eval_retrieval

## --- Interface (Phase 8) ---
app: ## Run the Streamlit chat UI
	$(PY) -m streamlit run app/streamlit_app.py

## --- Full stack ---
up: ## Start all services (docker-compose)
	$(COMPOSE) up -d

down: ## Stop all services (keeps data)
	$(COMPOSE) down

clean: ## Stop all services and DELETE data volumes (Qdrant/Postgres)
	$(COMPOSE) down -v
