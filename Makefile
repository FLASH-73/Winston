.PHONY: run install test lint format clean

run:
	cd winston && python main.py

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	mypy winston/

format:
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
