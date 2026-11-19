FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CORTEX_DATA_DIR=/data

WORKDIR /app

# Dependencies first for layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[llm,redis,dashboard]"

COPY config ./config
COPY examples ./examples
COPY dashboard ./dashboard

RUN useradd --create-home --uid 1000 cortex && mkdir -p /data && chown cortex /data
USER cortex
VOLUME ["/data"]

EXPOSE 8000
CMD ["cortex", "serve", "--host", "0.0.0.0", "--port", "8000"]
