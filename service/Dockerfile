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
    && apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY service ./service
COPY python/woundwatch ./python/woundwatch
COPY models/classifier.pt models/segmenter.pt ./models/
ENV PYTHONPATH=/app/python
RUN mkdir -p /app/runtime /var/data && chown -R appuser:appuser /app /var/data
EXPOSE 10000
CMD ["sh", "-c", "chown appuser:appuser /var/data && exec gosu appuser uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-10000} --no-access-log --proxy-headers --forwarded-allow-ips '*'"]
