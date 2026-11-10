.PHONY: help install test e2e mock api web build docker clean

help:
	@echo "JagX v13 — common tasks"
	@echo ""
	@echo "  make install   Install backend + frontend dependencies"
	@echo "  make test      Run the backend test suite (156 tests)"
	@echo "  make e2e       Run the end-to-end checks (needs a running API)"
	@echo "  make mock      Start the OpenAI-compatible mock LLM on :11434"
	@echo "  make api       Start the API on :8000"
	@echo "  make web       Start the frontend on :5173"
	@echo "  make build     Build the frontend for production"
	@echo "  make docker    Build and run with docker compose"
	@echo "  make clean     Remove caches and build output"

install:
	cd server && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	cd web && npm install

test:
	cd server && . .venv/bin/activate && python -m pytest -q

e2e:
	bash scripts/e2e_check.sh

mock:
	python3 scripts/mock_llm.py --port 11434

api:
	cd server && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000

web:
	cd web && npm run dev

build:
	cd web && npm run build

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf web/dist web/node_modules/.vite server/.pytest_cache
