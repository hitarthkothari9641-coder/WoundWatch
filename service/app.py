"""FastAPI service running the Python WoundWatch PyTorch checkpoints."""
import io,os
from fastapi import FastAPI,UploadFile,File,HTTPException
from PIL import Image,UnidentifiedImageError
import numpy as np
import torch
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from threading import Lock
import hashlib
from functools import lru_cache
from contextlib import asynccontextmanager
from .security import SecurityMiddleware, configuration
from woundwatch.config import CFG

from woundwatch.inference import load_classifier,load_segmenter,predict_image

MAX_BYTES=int(os.getenv("MAX_UPLOAD_BYTES",10*1024*1024)); MAX_PIXELS=int(os.getenv("MAX_IMAGE_PIXELS",20_000_000))
@asynccontextmanager
async def lifespan(app):
    configuration()
    if os.getenv('WOUNDWATCH_ENV')=='production':
        await run_in_threadpool(models_once)
    yield

app=FastAPI(title="WoundWatch research inference",version="2.1",lifespan=lifespan)
_models=None
_load_lock=Lock()

def model_paths():
    return {name:Path(os.getenv('WOUNDWATCH_'+name.upper()+'_PATH',str(CFG.models/f'{name}.pt')))
            for name in ('classifier','segmenter')}

@lru_cache(maxsize=1)
def model_fingerprints():
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in model_paths().items()}
def models_once():
    global _models
    with _load_lock:
        if _models is None:
            model_fingerprints()
            paths=model_paths()
            _models=(load_segmenter(paths['segmenter']),load_classifier(paths['classifier']))
    return _models

@app.get("/health")
def health():
    try: models_once()
    except Exception: raise HTTPException(503,"Model checkpoints unavailable")
    return {"status":"ok"}

@app.get('/live',include_in_schema=False)
def live(): return {'status':'alive'}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    data=await file.read(MAX_BYTES+1); await file.close()
    image=decode_image(data)
    def run():
        try:
            from .dashboard import inference_lock
            with inference_lock:
                (seg,_),(clf,ck)=models_once()
                return predict_image(image,seg,clf,ck)
        except Exception:
            raise HTTPException(503,"Inference failed; check model installation")
    return await run_in_threadpool(run)

def decode_image(data):
    if MAX_BYTES<1 or MAX_PIXELS<1: raise HTTPException(503,"Service limits misconfigured")
    if len(data)>MAX_BYTES: raise HTTPException(413,"Upload exceeds size limit")
    if not data: raise HTTPException(400,"Empty upload")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}: raise HTTPException(415,"Only JPEG and PNG images are supported")
            if source.width*source.height>MAX_PIXELS: raise HTTPException(413,"Image dimensions exceed limit")
            source.load(); image=Image.fromarray(np.asarray(source.convert("RGB"),dtype=np.uint8))
    except HTTPException: raise
    except Image.DecompressionBombError: raise HTTPException(413,"Image dimensions exceed limit")
    except (UnidentifiedImageError,OSError,ValueError): raise HTTPException(400,"Unreadable image")
    return image

from .dashboard import router
app.include_router(router)
WEB = Path(__file__).resolve().parent / 'web'
app.mount('/assets', StaticFiles(directory=WEB), name='assets')

@app.get('/', include_in_schema=False)
def dashboard():
    return FileResponse(WEB / 'index.html')

@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return FileResponse(WEB / 'favicon.svg', media_type='image/svg+xml')

@app.middleware('http')
async def private_responses(request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; frame-ancestors 'none'"
    return response

app.add_middleware(SecurityMiddleware)
