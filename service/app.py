# service/app.py
"""Small ONNX Runtime API for WoundWatch.

Input contract (matching code/predict_ssi.m): decode stored RGB pixels without
EXIF orientation transforms; make three-channel RGB; resize to the trained
network dimensions with bicubic interpolation; quantize resize output to
8-bit RGB; cast to float32 in [0,255], with no /255 or mean/std normalization.
The ONNX graph owns any normalization embedded during MATLAB export. Class
output order is [Low, Elevated]. Calibration is sigmoid(intercept + slope *
logit(Elevated score)) using the MATLAB Platt coefficients.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(20_000_000)))
CLASSIFIER_ONNX = os.getenv("CLASSIFIER_ONNX", "/models/classifier.onnx")
SEGMENTER_ONNX = os.getenv("SEGMENTER_ONNX", "/models/segmenter.onnx")


def _required_float(name: str) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Required deployment setting {name} is not configured")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"Deployment setting {name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise RuntimeError(f"Deployment setting {name} must be finite")
    return parsed


@dataclass(frozen=True)
class DeploymentSettings:
    platt_intercept: float
    platt_slope: float
    threshold: float
    defer_half_width: float
    blur_variance_minimum: float
    max_clipped_fraction: float
    min_wound_area_fraction: float
    max_wound_area_fraction: float


def load_settings() -> DeploymentSettings:
    """Read values copied from MATLAB config/model MAT files at deployment."""
    result = DeploymentSettings(
        platt_intercept=_required_float("PLATT_INTERCEPT"),
        platt_slope=_required_float("PLATT_SLOPE"),
        threshold=_required_float("DECISION_THRESHOLD"),
        defer_half_width=float(os.getenv("DEFER_HALF_WIDTH", "0.05")),
        blur_variance_minimum=float(os.getenv("BLUR_VARIANCE_MINIMUM", "0.0001")),
        max_clipped_fraction=float(os.getenv("MAX_CLIPPED_FRACTION", "0.05")),
        min_wound_area_fraction=float(os.getenv("MIN_WOUND_AREA_FRACTION", "0.002")),
        max_wound_area_fraction=float(os.getenv("MAX_WOUND_AREA_FRACTION", "0.80")),
    )
    if not 0.0 <= result.threshold <= 1.0:
        raise RuntimeError("DECISION_THRESHOLD must be in [0,1]")
    if not 0.0 <= result.defer_half_width <= 1.0:
        raise RuntimeError("DEFER_HALF_WIDTH must be in [0,1]")
    if not 0.0 <= result.max_clipped_fraction <= 1.0:
        raise RuntimeError("MAX_CLIPPED_FRACTION must be in [0,1]")
    if not 0.0 <= result.min_wound_area_fraction <= result.max_wound_area_fraction <= 1.0:
        raise RuntimeError("Wound-area fractions must satisfy 0 <= min <= max <= 1")
    return result


@dataclass
class Runtime:
    classifier: ort.InferenceSession
    segmenter: ort.InferenceSession
    settings: DeploymentSettings


_runtime: Runtime | None = None
app = FastAPI(title="WoundWatch inference", version="1.0.0")


def _load_runtime() -> Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime
    settings = load_settings()
    for model_path in (CLASSIFIER_ONNX, SEGMENTER_ONNX):
        if not os.path.isfile(model_path):
            raise RuntimeError("Required ONNX model file is missing")
    options = ort.SessionOptions()
    # Keep inference CPU-only and bound thread fan-out in small service containers.
    options.intra_op_num_threads = max(1, int(os.getenv("ONNX_THREADS", "1")))
    options.inter_op_num_threads = 1
    classifier = ort.InferenceSession(CLASSIFIER_ONNX, sess_options=options,
                                     providers=["CPUExecutionProvider"])
    segmenter = ort.InferenceSession(SEGMENTER_ONNX, sess_options=options,
                                    providers=["CPUExecutionProvider"])
    if len(classifier.get_inputs()) != 1 or len(segmenter.get_inputs()) != 1:
        raise RuntimeError("Expected one image input in each ONNX model")
    _runtime = Runtime(classifier, segmenter, settings)
    return _runtime


def _model_hw(session: ort.InferenceSession, fallback: tuple[int, int]) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    # Exported MATLAB image graphs normally expose NCHW; also accept NHWC.
    dims = list(shape)
    if len(dims) != 4:
        raise RuntimeError("Expected a four-dimensional ONNX image input")
    def known_int(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, np.integer)) and int(value) > 0 else None
    if known_int(dims[1]) == 3:
        return known_int(dims[2]) or fallback[0], known_int(dims[3]) or fallback[1]
    if known_int(dims[3]) == 3:
        return known_int(dims[1]) or fallback[0], known_int(dims[2]) or fallback[1]
    # Dynamic channel axes are ambiguous; use the explicitly documented layout.
    layout = os.getenv("ONNX_IMAGE_LAYOUT", "NCHW").upper()
    if layout == "NCHW":
        return known_int(dims[2]) or fallback[0], known_int(dims[3]) or fallback[1]
    if layout == "NHWC":
        return known_int(dims[1]) or fallback[0], known_int(dims[2]) or fallback[1]
    raise RuntimeError("ONNX_IMAGE_LAYOUT must be NCHW or NHWC")


def _prepare_rgb(image: Image.Image, height: int, width: int) -> np.ndarray:
    # MATLAB imread uses stored pixels; do not apply EXIF orientation. convert()
    # makes a new RGB image with no EXIF payload. MATLAB imresize uses bicubic.
    rgb = image.convert("RGB")
    resized = rgb.resize((width, height), resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.uint8)
    return pixels.astype(np.float32)  # Preserve MATLAB single(uint8), values 0..255.


def _format_input(session: ort.InferenceSession, pixels: np.ndarray) -> np.ndarray:
    dims = session.get_inputs()[0].shape
    if len(dims) != 4:
        raise RuntimeError("Expected a four-dimensional ONNX image input")
    if isinstance(dims[1], (int, np.integer)) and int(dims[1]) == 3:
        return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
    if isinstance(dims[3], (int, np.integer)) and int(dims[3]) == 3:
        return pixels[None, ...].astype(np.float32, copy=False)
    layout = os.getenv("ONNX_IMAGE_LAYOUT", "NCHW").upper()
    if layout == "NCHW":
        return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
    if layout == "NHWC":
        return pixels[None, ...].astype(np.float32, copy=False)
    raise RuntimeError("ONNX_IMAGE_LAYOUT must be NCHW or NHWC")


def _gray01(rgb: np.ndarray) -> np.ndarray:
    # Equivalent coefficients to MATLAB rgb2gray for normalized sRGB values.
    arr = rgb.astype(np.float64) / 255.0
    return 0.2989360213 * arr[..., 0] + 0.5870430745 * arr[..., 1] + 0.1140209043 * arr[..., 2]


def _quality_reasons(original_rgb: np.ndarray, runtime: Runtime) -> list[str]:
    gray = _gray01(original_rgb)
    # fspecial("laplacian",0.2) coefficients used by MATLAB quality_check.m.
    laplacian = np.array([[1 / 6, 2 / 3, 1 / 6],
                          [2 / 3, -10 / 3, 2 / 3],
                          [1 / 6, 2 / 3, 1 / 6]], dtype=np.float64)
    # MATLAB imfilter(...,"replicate","conv"): edge-value padding and
    # convolution. The kernel is symmetric, so correlation is equivalent.
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    response = sum(laplacian[r, c] * padded[r:r + gray.shape[0], c:c + gray.shape[1]]
                   for r in range(3) for c in range(3))
    settings = runtime.settings
    reasons: list[str] = []
    if float(np.var(response, ddof=1)) < settings.blur_variance_minimum:
        reasons.append("Image is too blurry (variance of Laplacian below threshold).")
    clipping = float(np.mean(gray <= 0.01) + np.mean(gray >= 0.99))
    if clipping > settings.max_clipped_fraction:
        reasons.append("Exposure has excessive dark or bright histogram clipping.")

    session = runtime.segmenter
    height, width = _model_hw(session, (256, 256))
    pixels = _prepare_rgb(Image.fromarray(original_rgb), height, width)
    feed = _format_input(session, pixels)
    output = session.run(None, {session.get_inputs()[0].name: feed})[0]
    prediction = np.asarray(output)
    if prediction.ndim != 4:
        raise RuntimeError("Segmenter output must be a four-dimensional image tensor")
    if prediction.shape[1] == 2:  # NCHW classes
        wound = prediction[0, 1] >= prediction[0, 0]
    elif prediction.shape[-1] == 2:  # NHWC classes
        wound = prediction[0, ..., 1] >= prediction[0, ..., 0]
    else:
        raise RuntimeError("Segmenter output must contain background and wound scores")
    wound_at_original_size = Image.fromarray(wound.astype(np.uint8) * 255).resize(
        (original_rgb.shape[1], original_rgb.shape[0]), resample=Image.Resampling.NEAREST)
    area = float(np.mean(np.asarray(wound_at_original_size, dtype=np.uint8) != 0))
    if area < settings.min_wound_area_fraction:
        reasons.append("Predicted wound area is below the configured minimum.")
    elif area > settings.max_wound_area_fraction:
        reasons.append("Predicted wound area exceeds the configured maximum.")
    return reasons


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _elevated_score(runtime: Runtime, rgb: np.ndarray) -> float:
    session = runtime.classifier
    height, width = _model_hw(session, (224, 224))
    pixels = _prepare_rgb(Image.fromarray(rgb), height, width)
    output = np.asarray(session.run(None, {session.get_inputs()[0].name: _format_input(session, pixels)})[0])
    values = output.reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("Classifier must return two finite outputs in [Low, Elevated] order")
    if np.any(values < -1e-5) or np.any(values > 1.00001):
        raise RuntimeError("Classifier outputs are not probabilities; export a probability-output model")
    if not np.isclose(float(np.sum(values)), 1.0, atol=1e-3):
        raise RuntimeError("Classifier output probabilities do not sum to one")
    raw = float(np.clip(values[1], 1e-7, 1.0 - 1e-7))
    logit = math.log(raw / (1.0 - raw))
    return _sigmoid(runtime.settings.platt_intercept + runtime.settings.platt_slope * logit)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _load_runtime()
    except Exception:
        # Do not expose filesystem paths, exception details, or environment values.
        raise HTTPException(status_code=503, detail="Models or inference settings are unavailable")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if MAX_UPLOAD_BYTES < 1 or MAX_IMAGE_PIXELS < 1:
        raise HTTPException(status_code=503, detail="Service limits are misconfigured")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await file.read(min(1024 * 1024, MAX_UPLOAD_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds upload size limit")
            chunks.append(chunk)
    finally:
        await file.close()
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    try:
        with Image.open(io.BytesIO(b"".join(chunks))) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed limit")
            if source.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
                raise HTTPException(status_code=400, detail="Only 8-bit images are supported")
            source.load()
            # Conversion drops metadata. Stored pixel orientation is preserved,
            # matching MATLAB imread behavior used by the training/inference code.
            image = Image.fromarray(np.asarray(source.convert("RGB"), dtype=np.uint8), mode="RGB")
            rgb = np.asarray(image, dtype=np.uint8)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Upload is not a supported readable image")
    try:
        runtime = _load_runtime()
        reasons = _quality_reasons(rgb, runtime)
        if reasons:
            return {"status": "retake", "probability": None, "risk_band": "unknown", "reasons": reasons}
        probability = _elevated_score(runtime, rgb)
        if abs(probability - runtime.settings.threshold) <= runtime.settings.defer_half_width:
            return {"status": "defer", "probability": probability, "risk_band": "defer",
                    "reasons": ["Calibrated probability is within the configured defer band around the decision threshold."]}
        band = "elevated" if probability >= runtime.settings.threshold else "low"
        return {"status": "ok", "probability": probability, "risk_band": band, "reasons": []}
    except HTTPException:
        raise
    except Exception:
        # Never log image bytes, filenames, or raw model exception details.
        raise HTTPException(status_code=503, detail="Inference failed; check service model configuration")
