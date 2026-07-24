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
        eval-llm app monitoring-up monitoring-down grafana \
        build up down clean logs ps

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

eval-llm: ## Compare prompt variants with an LLM-as-judge
	$(PY) -m eval.eval_llm

## --- Interface (Phase 8) ---
app: ## Run the Streamlit chat UI
	$(PY) -m streamlit run app/streamlit_app.py

## --- Monitoring (Phase 9): Postgres + Grafana ---
monitoring-up: ## Start Postgres + Grafana (provisioned dashboard)
	$(COMPOSE) up -d postgres grafana

monitoring-down: ## Stop Postgres + Grafana (keeps data)
	$(COMPOSE) stop postgres grafana

grafana: ## Print the Grafana dashboard URL
	@echo "Open http://localhost:$${GRAFANA_PORT:-3000} (login: admin / admin) -> dashboard 'Valluvan — Monitoring'"

## --- Full stack ---
build: ## Build the app image (Streamlit UI + ingest share it)
	$(COMPOSE) build

up: ## Build + start the WHOLE stack (qdrant, postgres, grafana, ingest, app)
	$(COMPOSE) up -d --build
	@echo "Stack starting. App -> http://localhost:$${APP_PORT:-8501}  |  Grafana -> http://localhost:$${GRAFANA_PORT:-3000}"

logs: ## Tail logs for all services (Ctrl-C to stop)
	$(COMPOSE) logs -f

ps: ## Show status of all services
	$(COMPOSE) ps

down: ## Stop all services (keeps data)
	$(COMPOSE) down

clean: ## Stop all services and DELETE data volumes (Qdrant/Postgres/models)
	$(COMPOSE) down -v
