FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/python WOUNDWATCH_ENV=production WOUNDWATCH_DATA_DIR=/tmp/runtime \
    WOUNDWATCH_TORCH_THREADS=2 MAX_IMAGE_PIXELS=4000000
WORKDIR /app
COPY service/requirements.txt ./requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt "httpx>=0.27,<0.29" \
    && useradd -m -u 1000 user
COPY --chown=user service ./service
COPY --chown=user python/woundwatch ./python/woundwatch
COPY --chown=user models ./models
USER user
# Bake ImageNet weights for the ensemble members into the image so cold starts do not download ~1 GB.
RUN python -c "from woundwatch.ensemble import build_backbone; [build_backbone(n) for n in ['resnet18','resnet50','resnet101','convnext_tiny','convnext_small','efficientnet_b0','efficientnet_b3','densenet121','swin_t','vit_b_16','regnet_y_8gf']]"
EXPOSE 7860
# WOUNDWATCH_ALLOWED_HOSTS and WOUNDWATCH_ORIGIN_SECRET come from Space settings (Variables / Secrets).
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","7860","--no-access-log","--proxy-headers","--forwarded-allow-ips","*"]
