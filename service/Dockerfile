# service/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY service ./service
COPY python/woundwatch ./python/woundwatch
ENV PYTHONPATH=/app/python
RUN mkdir -p /app/models /app/runtime && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
