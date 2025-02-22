.PHONY: setup clean test lint run

# Setup environment
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cp .env.template .env

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .tox/
	rm -rf data/processed/*
	rm -rf data/models/*

# Run tests
test:
	pytest tests/ -v --cov=src

# Run linting
lint:
	black src/ tests/
	flake8 src/ tests/
	mypy src/ tests/

# Run the pipeline
run:
	python src/main.py

# Create necessary directories
init:
	mkdir -p data/raw data/processed data/models logs

# Install development dependencies
dev-setup: setup
	pip install black flake8 mypy pytest pytest-cov

# Run all checks before commit
pre-commit: lint test