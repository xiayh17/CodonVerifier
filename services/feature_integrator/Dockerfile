FROM python:3.10-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir numpy pandas

# Copy service code
COPY services/feature_integrator/ /app/services/feature_integrator/

# Set entrypoint
ENTRYPOINT ["python", "/app/services/feature_integrator/app.py"]
