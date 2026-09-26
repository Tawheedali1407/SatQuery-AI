# ---- 1. build the React console ----
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python API that also serves the built console ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
# Build with `--build-arg WITH_ML=1` to include ChangeFormer (adds ~1 GB for CPU torch + 165 MB weights).
ARG WITH_ML=0
COPY backend/requirements*.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt && \
    if [ "$WITH_ML" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir timm einops transformers; \
    fi
COPY backend/ backend/
RUN if [ "$WITH_ML" = "1" ]; then python backend/scripts/fetch_models.py; fi
COPY --from=web /web/dist frontend/dist
# Demo scenes are downloaded at build time (public sources, see scripts/fetch_samples.py).
# Failures are tolerated so the image still builds offline.
RUN python backend/scripts/fetch_samples.py || true
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
