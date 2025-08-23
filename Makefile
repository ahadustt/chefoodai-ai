# ChefoodAI AI Service - Development & Deployment Makefile

.PHONY: help install dev test lint format build run clean docker-build docker-run deploy

# Default target
help:
	@echo "ChefoodAI AI Service - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  type-check  - Run mypy type checking"
	@echo ""
	@echo "Application:"
	@echo "  run         - Run the application locally"
	@echo "  run-dev     - Run with development settings"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build production Docker image"
	@echo "  docker-dev   - Build development Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-test  - Run tests in Docker"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy      - Deploy to Google Cloud Run"
	@echo "  deploy-dev  - Deploy to development environment"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean       - Clean build artifacts"
	@echo "  security    - Run security checks"

# Python and dependency management
install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev: install
	pip install -e .
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy bandit

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=api --cov=config --cov-report=html --cov-report=term

test-integration:
	pytest tests/ -v -m integration

test-unit:
	pytest tests/ -v -m "not integration"

# Code quality
lint:
	flake8 api config tests
	mypy api config
	bandit -r api config -f json -o bandit-report.json

format:
	black api config tests
	isort api config tests

type-check:
	mypy api config

security:
	bandit -r api config
	pip-audit

# Application
run:
	python main.py

run-dev:
	ENVIRONMENT=development python main.py

run-prod:
	ENVIRONMENT=production python main.py

# Docker
docker-build:
	docker build -t chefoodai-ai-service:latest --target production .

docker-dev:
	docker build -t chefoodai-ai-service:dev --target development .

docker-run:
	docker run -p 8000:8000 -e GOOGLE_CLOUD_PROJECT=mychef-467404 chefoodai-ai-service:latest

docker-test:
	docker build -t chefoodai-ai-service:test --target development .
	docker run --rm chefoodai-ai-service:test pytest tests/ -v

docker-shell:
	docker run -it --rm chefoodai-ai-service:latest /bin/bash

# Docker Compose (if needed)
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

# Deployment
deploy:
	gcloud run deploy chefoodai-ai-service \
		--image gcr.io/mychef-467404/chefoodai-ai-service:latest \
		--region us-central1 \
		--platform managed \
		--allow-unauthenticated \
		--min-instances 1 \
		--max-instances 5 \
		--cpu 4 \
		--memory 2Gi \
		--timeout 600 \
		--set-env-vars ENVIRONMENT=production

deploy-dev:
	gcloud run deploy chefoodai-ai-service-dev \
		--image gcr.io/mychef-467404/chefoodai-ai-service:dev \
		--region us-central1 \
		--platform managed \
		--allow-unauthenticated \
		--min-instances 0 \
		--max-instances 2 \
		--cpu 2 \
		--memory 1Gi \
		--set-env-vars ENVIRONMENT=development

# Build and push to Google Container Registry
docker-push:
	docker tag chefoodai-ai-service:latest gcr.io/mychef-467404/chefoodai-ai-service:latest
	docker push gcr.io/mychef-467404/chefoodai-ai-service:latest

# Full CI/CD pipeline simulation
ci: clean install lint test security
	@echo "All CI checks passed!"

cd: ci docker-build docker-push deploy
	@echo "Deployment completed!"

# Maintenance
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -f bandit-report.json
	rm -f coverage.xml

clean-docker:
	docker system prune -f
	docker image prune -f

# Environment setup
setup-gcp:
	gcloud auth configure-docker
	gcloud config set project mychef-467404
	gcloud services enable run.googleapis.com
	gcloud services enable cloudbuild.googleapis.com

# Database migrations (if needed in future)
migrate:
	@echo "No migrations needed for current setup"

# Monitoring and logs
logs:
	gcloud run services logs read chefoodai-ai-service --region=us-central1 --limit=50

logs-tail:
	gcloud run services logs tail chefoodai-ai-service --region=us-central1

# Performance testing
perf-test:
	@echo "Performance testing not yet implemented"

# Health checks
health-check:
	curl -f http://localhost:8000/health || echo "Service not running"

# Version management
version:
	@python -c "from config import get_settings; print(f'Version: {get_settings().app_version}')"

# Environment validation
validate-env:
	@python -c "from config import get_settings, validate_production_config; settings = get_settings(); issues = validate_production_config(settings); print('✅ Configuration valid' if not issues else f'❌ Issues: {issues}')"