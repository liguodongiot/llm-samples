FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/
COPY docs/ docs/

RUN pip install --no-cache-dir pytest pytest-cov

#CMD ["python", "examples/basic_usage.py"]