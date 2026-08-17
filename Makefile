# Fraud Detection API — developer commands.
# `make up` builds and runs the API, background worker, and Redis in Docker.
#
# NOTE: this app runs its ML models in-process (HuggingFace transformers), so it
# has no Ollama service and no `ensure-ollama-network` target — it does not use
# the shared word-games-ollama daemon. Redis is its only datastore.

.PHONY: up down restart logs ps rebuild sh check-env init-env \
	install-dev format format-check lint type-check test check clean help

COMPOSE = docker compose --env-file .env
LOG_TAIL ?= 200

help:
	@echo "Stack:   make up | down | restart | logs | ps | rebuild | sh"
	@echo "Env:     make check-env | init-env"
	@echo "Quality: make format | format-check | lint | type-check | test | check"
	@echo "         make install-dev | clean"

# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

## Start the API + worker + Redis in the background (builds if needed).
up:
	chmod 600 .env
	$(MAKE) check-env
	$(COMPOSE) up -d --build

## Stop and remove the containers. Named volumes survive; `down -v` clears them.
down:
	$(COMPOSE) down

## Restart the API container.
restart:
	$(COMPOSE) restart app

## Follow API logs. Override history with LOG_TAIL=500 or LOG_TAIL=all.
logs:
	$(COMPOSE) logs --follow --tail=$(LOG_TAIL) app

## Show container status.
ps:
	$(COMPOSE) ps

## Rebuild the API image from scratch (no cache).
rebuild:
	$(COMPOSE) build --no-cache app

## Open a shell in the API container.
sh:
	$(COMPOSE) exec app sh

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

## .env is the only environment file allowed anywhere in this repo, it must
## be mode 0600, and it must carry exactly one entry for every key init-env
## emits — so a variable the code starts reading can never be silently absent.
check-env:
	@test -f .env || (echo "check-env: .env is missing; run 'make init-env'" >&2; exit 1)
	@extra=$$(find . -name '.env' -o -name '.env.*' 2>/dev/null \
		| grep -Ev '(^|/)(node_modules|\.git|\.venv|venv|\.next|\.claude)/' \
		| grep -v '^\./.env$$' || true); \
	if [ -n "$$extra" ]; then \
		echo "check-env: only .env is allowed; remove:" >&2; echo "$$extra" | sed 's/^/  /' >&2; exit 1; \
	fi
	@mode=$$(stat -c '%a' .env 2>/dev/null || stat -f '%Lp' .env); \
	if [ "$$mode" != "600" ]; then \
		echo "check-env: .env permissions are $$mode; expected 600" >&2; exit 1; \
	fi
	@bad=$$(grep -oE "^[[:space:]]+['\"][A-Z][A-Z0-9_]*=" Makefile | grep -oE "[A-Z][A-Z0-9_]*" | sort -u \
		| while read -r key; do \
			[ "$$(grep -c "^$$key=" .env)" -eq 1 ] || echo "  $$key"; \
		done); \
	if [ -n "$$bad" ]; then \
		echo "check-env: .env needs exactly one entry per init-env key; missing or duplicated:" >&2; \
		echo "$$bad" >&2; exit 1; \
	fi
	@echo "check-env: clean — .env is complete and mode 0600"

## Create the one canonical .env with safe local defaults (only if missing).
init-env:
	@if [ -f .env ]; then \
		echo "init-env: .env already exists; leaving it untouched"; \
	else \
		printf '%s\n' \
			'ENVIRONMENT=development' \
			'DETECT_FRAUD_SECURITY_KEY=change-me-local-detect-key' \
			'FINETUNE_MODEL_SECURITY_KEY=change-me-local-finetune-key' \
			'MAX_REQUEST_BODY_BYTES=1048576' \
			'REQUEST_TIMEOUT_SECONDS=60' \
			'RATE_LIMIT_REQUESTS=30' \
			'RATE_LIMIT_WINDOW_SECONDS=60' \
			'MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
			'TRAIN_BATCH_SIZE=1' 'TRAIN_EPOCHS=1' 'TRAIN_MAX_STEPS=100' \
			'DATA_TRAIN_END=1000' 'LOW_CPU_MEM_USAGE=True' 'MAX_SEQ_LENGTH=256' \
			'HF_HOME=/app/models/hf' \
			'REDIS_HOST=redis' 'REDIS_PORT=6379' 'REDIS_DB=0' \
			'REDIS_HOST_PORT=6379' 'API_HOST_PORT=8899' \
			'LOG_LEVEL=INFO' 'LOG_DIR=/app/logs' > .env; \
		chmod 600 .env; \
		echo "init-env: wrote .env with safe local defaults"; \
	fi

# ---------------------------------------------------------------------------
# Quality / tests
# ---------------------------------------------------------------------------

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

format:
	@echo "Running isort..."
	isort app/
	@echo "Running black..."
	black app/

format-check:
	@echo "Checking isort..."
	isort --check-only app/
	@echo "Checking black..."
	black --check app/

lint:
	@echo "Running flake8..."
	flake8 app/

type-check:
	@echo "Running mypy..."
	mypy app/

test:
	@echo "Running pytest..."
	pytest

check: format-check lint type-check test
	@echo "All checks passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage build/ dist/
