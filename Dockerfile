FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies (build-essential for C extensions, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, configuration, assets, and serialized models
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY .streamlit/ ./.streamlit/
COPY app.py .
COPY MODEL_CARD.md .
COPY PROJECT_AUDIT.md .
COPY README.md .

# Create non-root user for production security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose default port
EXPOSE 8501

# Healthcheck targeting Streamlit core healthcheck endpoint using dynamic PORT
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Launch Streamlit with dynamic PORT and 0.0.0.0 binding
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
