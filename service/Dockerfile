# service/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY service/app.py ./service/app.py
COPY python/woundwatch ./python/woundwatch
ENV PYTHONPATH=/app/python
RUN mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
