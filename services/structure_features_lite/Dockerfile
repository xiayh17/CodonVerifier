FROM python:3.10-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir numpy pandas

# Copy service code
COPY services/structure_features_lite/ /app/services/structure_features_lite/

# Set entrypoint
ENTRYPOINT ["python", "/app/services/structure_features_lite/app.py"]
