# Evo2 Service - DNA Sequence Generation
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Evo2 dependencies
RUN pip install --no-cache-dir \
    biopython \
    "huggingface_hub>=0.22,<0.36" \
    einops \
    safetensors \
    sentencepiece \
    tqdm

# Install Evo2
RUN pip install --no-cache-dir evo2 || pip install --no-deps evo2

# Verify installation
RUN python3 -c "import torch; import evo2; print('✓ Evo2 service ready')"

# Copy service application
COPY services/evo2/app_enhanced.py /app/
COPY services/evo2/utils.py /app/
COPY codon_verifier/ /app/codon_verifier/

# Create cache directory
RUN mkdir -p /cache/huggingface

ENTRYPOINT ["python3", "app_enhanced.py"]
CMD ["--help"]

