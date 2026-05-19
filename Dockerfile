# Versions testées : Docker 24+, image python:3.11-slim
FROM python:3.11-slim

# Installer curl
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Créer l'utilisateur AVANT toute utilisation
RUN useradd -m appuser

WORKDIR /app

# Dépendances d'abord (cache des couches)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Code applicatif
COPY app/ ./app/
COPY model/ ./model/

# Donner les droits à l'utilisateur
RUN chown -R appuser:appuser /app

# Passer en non-root
USER appuser

EXPOSE 8000

# Check model every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]