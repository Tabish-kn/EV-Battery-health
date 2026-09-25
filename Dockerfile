FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and serialized artifacts
COPY feature_engine.py .
COPY eda_and_model.py .
COPY app.py .
COPY ev_pipeline_bundle.pkl .
COPY ev_battery_model.pkl .
COPY scaler.pkl .
COPY imputer.pkl .
COPY feature_columns.pkl .
COPY ev_battery_health.csv .

# Expose default Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit with headless settings
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
