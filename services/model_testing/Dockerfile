# Model Testing Service - Production Model Testing and Evaluation
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies for testing
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    biopython \
    tqdm \
    lightgbm \
    joblib \
    matplotlib \
    seaborn

# Copy the codon_verifier module
COPY codon_verifier /app/codon_verifier

# Copy scripts needed for testing
COPY scripts/test_production_model.py /app/scripts/
COPY scripts/train_with_all_features.py /app/scripts/

# Copy service application
COPY services/model_testing/app.py /app/

# Create output directories
RUN mkdir -p /data/test_results /data/plots

# Verify installation
RUN python3 -c "import sklearn; import numpy; import pandas; import matplotlib; print('✓ Model testing service ready')"

ENTRYPOINT ["python3", "app.py"]
CMD ["--help"]
