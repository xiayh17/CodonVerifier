FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for MMseqs2
RUN apt-get update && apt-get install -y \
    wget \
    cmake \
    g++ \
    zlib1g-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2
RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && \
    tar xvfz mmseqs-linux-avx2.tar.gz && \
    mv mmseqs/bin/mmseqs /usr/local/bin/ && \
    rm -rf mmseqs mmseqs-linux-avx2.tar.gz

# Install Python dependencies
RUN pip install --no-cache-dir numpy pandas

# Copy service code
COPY services/msa_features_lite/ /app/services/msa_features_lite/

# Set working directory
WORKDIR /app/services/msa_features_lite

# Set default command (can be overridden)
CMD ["python", "app.py"]
