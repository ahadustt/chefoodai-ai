# Multi-stage production-optimized build for ChefoodAI AI Service
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Build stage - Install dependencies and build wheels
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with proper permissions
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /tmp/uploads && \
    chown -R appuser:appuser /app /tmp/uploads

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Create .env file with safe defaults for production
RUN echo "ENVIRONMENT=production" > .env && \
    echo "LOG_LEVEL=INFO" >> .env && \
    echo "ENABLE_METRICS=true" >> .env && \
    chown appuser:appuser .env

# Switch to non-root user
USER appuser

# Set default port (Cloud Run will override this)
ENV PORT=8000

# Expose port (documentation only)
EXPOSE 8000

# Health check with proper timeout and retries
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the application using the optimized startup script
CMD ["python", "startup.py"]

# Multi-target support for development
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN /opt/venv/bin/pip install --no-cache-dir \
    pytest>=7.4.3 \
    pytest-asyncio>=0.21.1 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    flake8>=6.0.0

# Switch back to app user
USER appuser

# Override for development
ENV ENVIRONMENT=development \
    LOG_LEVEL=DEBUG \
    ENABLE_METRICS=true

# Development command with auto-reload
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

