### Builder stage: build wheels for all requirements
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120

# Install build dependencies for packages that compile native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy requirements and build wheelhouse to allow caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip wheel --timeout 120 --retries 5 --no-cache-dir --no-deps -r requirements.txt -w /wheels


### Runtime stage: only runtime artifacts
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Minimal runtime deps (keep small). Many Python packages provide manylinux wheels,
# so compiling at runtime is not required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built wheels and install them
COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip && \
    pip install --timeout 120 --retries 5 --no-cache-dir /wheels/*.whl

# Copy application source
COPY src/ ./src/
COPY pyproject.toml ./

# Expose default port
EXPOSE 8080

# Run the app from the src package
WORKDIR /app/src
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
