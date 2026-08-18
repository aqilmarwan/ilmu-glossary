# Routing-aware calibration study.
#
# Local targets are for development and the free checks. The real runs go
# through Modal - see `make modal-run`.

CONFIG ?= configs/config.yaml
MODAL  := modal run modal_app/app.py

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Sync host dependencies (no CUDA)
	uv sync --extra dev

.PHONY: check
check: lint typecheck test ## Lint, typecheck and test

.PHONY: lint
lint: ## ruff check + format check
	uv run ruff check src/ tests/ modal_app/
	uv run ruff format --check src/ tests/ modal_app/

.PHONY: fmt
fmt: ## Apply ruff formatting and autofixes
	uv run ruff check --fix src/ tests/ modal_app/
	uv run ruff format src/ tests/ modal_app/

.PHONY: typecheck
typecheck: ## mypy strict
	uv run mypy src/ilmu_glossary

.PHONY: test
test: ## pytest
	uv run pytest -q

.PHONY: preflight
preflight: ## Validate model ids, datasets and recipes - costs nothing
	uv run ilmu preflight --config $(CONFIG)

.PHONY: preflight-offline
preflight-offline: ## Preflight without touching the Hub
	uv run ilmu preflight --config $(CONFIG) --no-check-hub

.PHONY: modal-dry
modal-dry: ## End-to-end smoke test on a small MoE, tiny N
	$(MODAL) --config $(CONFIG) --dry-run

.PHONY: modal-run
modal-run: preflight ## The staged run on B200 (honours the spec's gates)
	$(MODAL) --config $(CONFIG)

.PHONY: report
report: ## Regenerate REPORT.md from existing artifacts
	uv run ilmu report --config $(CONFIG)

.PHONY: clean
clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
