# Nigerian Transactions Fraud Detection API

A FastAPI-based fraud detection service for Nigerian transactions, featuring asynchronous job processing with Dramatiq workers and Redis queue management.

## Features

- **Asynchronous Job Processing**: Background job queue using Dramatiq and Redis
- **Fraud Detection**: ML-powered fraud detection for transaction analysis
- **Model Fine-tuning**: Endpoint for fine-tuning fraud detection models
- **Job Status Tracking**: Real-time job status monitoring via REST API
- **Structured Logging**: Comprehensive logging with middleware support
- **Production Ready**: Docker Compose setup for both development and production

## Model & Dataset

The fraud detection model (`Qwen/Qwen3-1.7B`) was trained using **5 million rows** of transaction data from the [Nigerian Financial Transactions and Fraud Detection Dataset](https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset) on Hugging Face.

This dataset contains:

- 5,000,000 synthetic Nigerian financial transactions
- 45+ advanced fraud detection features
- Coverage of 6 Nigerian geo-regions
- Multiple fraud types including account takeover, identity fraud, SIM swap fraud, and more
- Nigerian-specific payment channels (USSD, Mobile App, Card, Bank Transfer)
- Localized merchant categories and user personas

The dataset provides comprehensive features for training fraud detection models specifically tailored to Nigerian financial transaction patterns.

## Architecture

The application consists of three main components:

1. **API Service** (`app`): FastAPI application handling HTTP requests
2. **Worker Service** (`worker`): Dramatiq workers processing background jobs
3. **Redis**: Message broker and job queue backend

## Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Redis (or use Docker Compose)

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

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Model Configuration
MODEL_NAME=Qwen/Qwen3-1.7B
TRAIN_BATCH_SIZE=4
TRAIN_EPOCHS=3
LOW_CPU_MEM_USAGE=False
DATA_TRAIN_END=10

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

```

### Run with Docker (Development)

```bash
# Build and run all services
docker-compose up --build

```

The API will be available at `http://localhost:8000`

## API Endpoints

### Docs

- `GET /docs` - Displays all the APIs in this project

## Project Structure

```
fraud-detector/
├── app/
│   ├── config/          # Configuration and settings
│   ├── database/        # Database connections (Redis)
│   ├── middlewares/      # FastAPI middlewares
│   ├── queues/           # Dramatiq job queue setup
│   ├── routes/           # API route handlers
│   ├── schemas/          # Pydantic models
│   ├── utils/            # Utility functions
│   ├── workers/          # Background job workers
│   └── main.py           # FastAPI application entry point
├── models/               # Trained model checkpoints
├── logs/                 # Application logs
├── docker-compose.yml    # Development Docker setup
├── docker-compose.prod.yml  # Production Docker setup
├── Dockerfile            # Container image definition
├── Makefile              # Development commands
├── pyproject.toml        # Python project configuration
└── requirements.txt      # Production dependencies
```

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

## Logging

The application uses structured logging with the following log files:

- `logs/app.log` - General application logs
- `logs/errors.log` - Error logs

Logging middleware automatically logs all HTTP requests and responses.
