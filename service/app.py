"""FastAPI service running the Python WoundWatch PyTorch checkpoints."""
import io,os
from fastapi import FastAPI,UploadFile,File,HTTPException
from PIL import Image,UnidentifiedImageError
import numpy as np
import torch

from woundwatch.inference import load_classifier,load_segmenter,predict_image

MAX_BYTES=int(os.getenv("MAX_UPLOAD_BYTES",10*1024*1024)); MAX_PIXELS=int(os.getenv("MAX_IMAGE_PIXELS",20_000_000))
app=FastAPI(title="WoundWatch research inference",version="2.0")
_models=None
def models_once():
    global _models
    if _models is None: _models=(load_segmenter(),load_classifier())
    return _models

@app.get("/health")
def health():
    try: models_once()
    except Exception: raise HTTPException(503,"Model checkpoints unavailable")
    return {"status":"ok"}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    if MAX_BYTES<1 or MAX_PIXELS<1: raise HTTPException(503,"Service limits misconfigured")
    data=await file.read(MAX_BYTES+1); await file.close()
    if len(data)>MAX_BYTES: raise HTTPException(413,"Upload exceeds size limit")
    if not data: raise HTTPException(400,"Empty upload")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width*source.height>MAX_PIXELS: raise HTTPException(413,"Image dimensions exceed limit")
            source.load(); image=Image.fromarray(np.asarray(source.convert("RGB"),dtype=np.uint8))
    except HTTPException: raise
    except (UnidentifiedImageError,OSError,ValueError): raise HTTPException(400,"Unreadable image")
    try:
        (seg,_),(clf,ck)=models_once()
        return predict_image(image,seg,clf,ck)
    except Exception:
        # Avoid logging image bytes, filenames, or patient-associated metadata.
        raise HTTPException(503,"Inference failed; check model installation")
