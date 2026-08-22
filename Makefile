.PHONY: setup test lint format typecheck clean lab notebook run

setup:      ## Sync the environment (creates .venv, installs deps from uv.lock).
	uv sync --all-extras

test:       ## Run the test suite.
	uv run pytest

lint:       ## Lint with ruff.
	uv run ruff check src tests

format:     ## Format with black.
	uv run black src tests

typecheck:  ## Type-check with mypy.
	uv run mypy src

clean:      ## Remove caches and build artifacts.
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov dist *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

lab:        ## Launch Jupyter Lab.
	uv run invoke lab

notebook:   ## Launch classic Jupyter Notebook.
	uv run invoke notebook

run:        ## Run the application entry point.
	uv run python app/main.py
