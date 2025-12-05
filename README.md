# Fraud Detection API

A FastAPI-based fraud detection service.

## Setup

### Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (formatters, linters, testing tools)
pip install -r requirements-dev.txt
```

Or use the virtual environment:

```bash
source .venv/bin/activate
make install-dev
```

### Run with Docker

```bash
# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop
docker-compose down
```

## Development

### Code Formatting

This project uses **Black** for code formatting and **isort** for import sorting.

```bash
# Format all code
make format

# Or manually:
black app/
isort app/
```

### Linting

```bash
# Run flake8 linter
make lint

# Run type checker
make type-check
```

### Testing

```bash
# Run tests with coverage
make test
```

### Run All Checks

```bash
# Format check, lint, type-check, and test
make check
```

### Available Make Commands

- `make install-dev` - Install development dependencies
- `make format` - Format code with black and isort
- `make lint` - Run flake8 linter
- `make type-check` - Run mypy type checker
- `make test` - Run pytest tests
- `make check` - Run all checks
- `make clean` - Remove cache and build files

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /detect-fraud` - Fraud detection endpoint
- `POST /finetune-model` - Model fine-tuning endpoint

## Configuration

The following tools are configured:

- **Black** (v24.10.0) - Code formatter with 88 character line length
- **isort** (v5.13.2) - Import sorter, configured to work with Black
- **flake8** (v7.1.1) - Linter for style guide enforcement
- **mypy** (v1.13.0) - Static type checker
- **pytest** (v8.3.4) - Testing framework with coverage reporting

All configurations are in:
- `pyproject.toml` - Black, isort, mypy, pytest configs
- `.flake8` - Flake8 configuration
- `Makefile` - Convenient command shortcuts
