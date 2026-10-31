"""Per-image inference, Grad-CAM, colour heatmap and doctor-facing guidance for a set of photos.

    python -m woundwatch.batch_inference [--classifier models/classifier_v2.pt] [--out results/inference]

Writes <out>/<id>.json, <id>_gradcam.jpg, <id>_colour.jpg and <out>/index.csv. Scores from a
checkpoint fitted on an image are not held-out for that image; `heldout` in the index is true only
when --heldout-split names a split the checkpoint never saw.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .benchmark_classifier import resolve_paths
from .color_features import feature_row
from .common import read_rgb
from .config import CFG
from .guidance import attention_regions, build_guidance
from .inference import load_classifier, load_segmenter, predict_image, segment_wound, wound_and_ring
from .visualization import colour_heatmap, gradcam_values, overlay


def analyse(image, seg, clf, ck):
    result = predict_image(image, seg, clf, ck)
    view = image.copy(); view.thumbnail((768, 768))
    shares = cam_img = colour = colour_err = None
    mask = segment_wound(view, seg)
    ring_px = max(1, round(12 * max(view.size) / 256))
    try:
        colour = feature_row(view, mask, ring_px)
    except ValueError as exc:
        colour_err = str(exc)
    found = float(mask.mean()) >= CFG.minimum_wound_area_fraction
    if result["status"] != "retake":
        try:
            _, cam = gradcam_values(image, clf, ck)
            if found:
                wound, ring = wound_and_ring(view, mask, ring_px)
                shares = attention_regions(cam, wound, ring)
            cam_img = overlay(view, cam)
        except ValueError:
            pass
    guide = build_guidance(result, shares, colour, float(ck["threshold_youden"]), outline_found=found)
    return result, shares, colour, colour_err, cam_img, guide


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classifier", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=CFG.results / "inference")
    ap.add_argument("--heldout-split", default=None, help="manifest split unseen by the checkpoint")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.environ.setdefault("WOUNDWATCH_AREA_GATE", "advisory")  # report every image; see MODEL_CARD
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CFG.manifest)
    paths = resolve_paths(df)
    seg, _ = load_segmenter(); clf, ck = load_classifier(a.classifier)
    rows = []
    for i, (row, path) in enumerate(zip(df.itertuples(index=False), paths)):
        if a.limit and i >= a.limit: break
        key = Path(row.file).stem
        image = read_rgb(path)
        result, shares, colour, colour_err, cam_img, guide = analyse(image, seg, clf, ck)
        colour_heatmap(image).save(a.out / f"{key}_colour.jpg", quality=85)
        if cam_img is not None: cam_img.save(a.out / f"{key}_gradcam.jpg", quality=85)
        rec = dict(id=key, split=row.split, true_label=row.label, result=result, attention=shares,
                   colour=colour, colour_error=colour_err, guidance=guide,
                   gradcam_image=f"{key}_gradcam.jpg" if cam_img is not None else None,
                   colour_image=f"{key}_colour.jpg", classifier=str(a.classifier or "models/classifier.pt"),
                   heldout=bool(a.heldout_split and row.split == a.heldout_split))
        (a.out / f"{key}.json").write_text(json.dumps(rec, indent=1))
        rows.append(dict(id=key, split=row.split, true_label=row.label, status=result["status"],
                         probability=result["probability"], risk_band=result["risk_band"],
                         heldout=rec["heldout"],
                         att_wound=shares and shares["wound"], att_ring=shares and shares["ring"]))
        if i % 50 == 0: print(i, key, result["status"], flush=True)
    pd.DataFrame(rows).to_csv(a.out / "index.csv", index=False)
    print(f"Wrote {len(rows)} analyses to {a.out}")


if __name__ == "__main__":
    main()
