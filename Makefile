.PHONY: format lint type-check test install-dev clean help

help:
	@echo "Available commands:"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make type-check   - Run mypy type checker"
	@echo "  make test         - Run pytest tests"
	@echo "  make check        - Run all checks (format, lint, type-check, test)"
	@echo "  make clean        - Remove cache and build files"

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
