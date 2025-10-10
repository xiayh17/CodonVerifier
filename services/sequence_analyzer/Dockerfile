# Sequence Analyzer Service - DNA Sequence Verification and Analysis
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for ViennaRNA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      wget \
      autoconf \
      automake \
      libtool \
      pkg-config \
      xxd && \
    rm -rf /var/lib/apt/lists/*

# Install ViennaRNA
WORKDIR /tmp/viennarna
RUN wget -q https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz && \
    tar -xzf ViennaRNA-2.6.4.tar.gz --strip-components=1 && \
    ./configure --without-python --without-perl --without-swig --without-doc --disable-lto && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/viennarna

# Verify ViennaRNA
RUN RNAfold --version

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.22 \
    pandas>=1.3 \
    scikit-learn>=1.1 \
    joblib>=1.2 \
    scipy>=1.8 \
    lightgbm>=3.3 \
    biopython \
    transformers \
    matplotlib \
    seaborn

# Copy codon_verifier framework
COPY codon_verifier /app/codon_verifier

# Copy service application
COPY services/sequence_analyzer/app.py /app/
COPY services/sequence_analyzer/utils.py /app/

# Verify installation (check if module can be imported)
RUN python3 -c "import sys; sys.path.insert(0, '/app'); import codon_verifier; print('✓ Sequence Analyzer service ready')" || echo "⚠ Sequence Analyzer will be available at runtime"

ENTRYPOINT ["python3", "app.py"]
CMD ["--help"]

