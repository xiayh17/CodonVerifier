# Training Service - Surrogate Model Training
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

# Install Python dependencies for training
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    biopython \
    tqdm \
    lightgbm

# Copy the codon_verifier module
COPY codon_verifier /app/codon_verifier

# Copy service application
COPY services/training/app.py /app/
COPY services/training/utils.py /app/

# Verify installation
RUN python3 -c "import sklearn; import numpy; import pandas; print('✓ Training service ready')"

ENTRYPOINT ["python3", "app.py"]
CMD ["--help"]

