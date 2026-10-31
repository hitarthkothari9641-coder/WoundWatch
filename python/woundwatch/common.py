"""Common deterministic, image and model utilities."""
import random
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)

_THREAD_CONFIGURED = False
def configure_threads():
    global _THREAD_CONFIGURED
    if not _THREAD_CONFIGURED:
        torch.set_num_threads(max(1,min(int(os.getenv('WOUNDWATCH_TORCH_THREADS','2')),os.cpu_count() or 1)))
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        _THREAD_CONFIGURED = True

def seed_all(seed):
    configure_threads()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device():
    configure_threads()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_rgb(path):
    try:
        with Image.open(path) as im:
            im.load()
            return Image.fromarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {path}") from exc

def image_tensor(image, size):
    image = image.convert("RGB")
    h, w = size
    rgb = np.asarray(image.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return torch.from_numpy(((rgb-MEAN)/STD).transpose(2,0,1).copy())

def binary_mask(mask):
    a = np.asarray(mask)
    if a.ndim == 3: a = a[..., 0]
    if a.size == 0: raise ValueError("Mask has no pixels")
    return a > 0

def find_mask(image_path, label_dir):
    p = Path(label_dir) / Path(image_path).name
    if p.is_file(): return p
    found = list(Path(label_dir).glob(Path(image_path).stem + ".*"))
    if len(found) == 1: return found[0]
    raise FileNotFoundError(f"Mask missing for {Path(image_path).name}")
