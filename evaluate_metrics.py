"""Evaluate WoundWatch's Python classifier without retraining or editing originals.

COLAB (run from your WoundWatch repository folder):
    !python evaluate_metrics.py
    !python evaluate_metrics.py --saved-predictions

Default: run models/classifier_candidate.pt on the supplied test images.
--saved-predictions: recompute metrics from the candidate's saved test CSV;
this needs only Python's standard library and does NOT run model inference.
To inspect the active all-data model:
    !python evaluate_metrics.py --checkpoint models/classifier.pt
Its supplied-test scores are NOT held-out performance (it trained on all images).

FLOW:
* SurgWound labels: Low -> 0; Medium/High -> Elevated (1).
* RGB -> bicubic resize 224x224 -> ImageNet normalization -> ResNet-18.
* Active training: geometric augmentation, weighted cross-entropy, AdamW,
  stratified five-fold CV, Platt calibration, then final fit on ALL images.
* Candidate: frozen ImageNet ResNet-18 features, standardized logistic head,
  nested CV on train+val, Platt calibration and threshold selected on development.
* Evaluation: calibrated Elevated probability >= saved Youden threshold.
* The service also uses DeepLabV3+ with ResNet-18 for segmentation, a quality
  gate and a defer band. This file measures the CLASSIFIER on every selected
  image; it does not score segmentation or exclude retake/defer cases.

Precision, recall and F1 use Elevated as the positive class; accuracy uses
both classes. All four are percentages. Confusion matrix contains counts.
Patient independence is unknown because the supplied data has no patient IDs.
No files are written unless --output is supplied (a NEW JSON file only).
Dependencies for inference: torch, torchvision, numpy, Pillow (Colab has these).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No evaluation rows in {path}")
    return rows


def label_value(label):
    if label not in {"Low", "Medium", "High"}:
        raise ValueError(f"Unexpected label {label!r}; expected Low, Medium, High")
    return int(label != "Low")


def calculate_metrics(labels, probabilities, threshold):
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be nonempty and aligned")
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    counts = [[0, 0], [0, 0]]
    for label, probability in zip(labels, probabilities):
        if label not in (0, 1):
            raise ValueError("Binary labels must be 0 or 1")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilities must be finite and in [0, 1]")
        counts[label][int(probability >= threshold)] += 1
    tn, fp = counts[0]
    fn, tp = counts[1]
    # Undefined ratios are reported as null/N/A, not misleading zero scores.
    ratio = lambda numerator, denominator: (
        100.0 * numerator / denominator if denominator else None)
    return {
        "n": len(labels), "positive_class": "Elevated",
        "threshold": threshold,
        "accuracy_percent": ratio(tp + tn, len(labels)),
        "precision_percent": ratio(tp, tp + fp),
        "recall_percent": ratio(tp, tp + fn),
        "f1_percent": ratio(2 * tp, 2 * tp + fp + fn),
        "confusion_matrix": counts,
        "class_order": ["Low", "Elevated"],
    }


def saved_predictions(root):
    rows = read_rows(root / "results/classifier_candidate_test_predictions.csv")
    benchmark = json.loads(
        (root / "results/classifier_candidate_benchmark.json").read_text(encoding="utf-8"))
    threshold = float(benchmark["heldout_test"]["youden_operating_point"]
                      ["threshold_from_development"])
    labels = [label_value(row["label"]) for row in rows]
    probabilities = [float(row["calibrated_probability"]) for row in rows]
    return labels, probabilities, threshold


def infer(root, checkpoint, manifest, split, batch_size):
    import numpy as np
    import torch
    from torch import nn
    from torchvision import models

    sys.path.insert(0, str(root / "python"))
    from woundwatch.common import device, image_tensor, read_rgb

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("backbone") != "resnet18" or ck.get("class_order") != ["Low", "Elevated"]:
        raise ValueError("Expected WoundWatch ResNet-18 checkpoint with Low/Elevated class order")
    if ck.get("imagenet_normalization") is not True:
        raise ValueError("Checkpoint does not specify expected ImageNet normalization")
    threshold = float(ck["threshold_youden"])
    intercept, slope = float(ck["platt_intercept"]), float(ck["platt_slope"])
    if not all(math.isfinite(v) for v in (intercept, slope)):
        raise ValueError("Invalid Platt calibration parameters")
    rows = read_rows(manifest)
    if not {"file", "split", "label"} <= rows[0].keys():
        raise ValueError("Manifest must contain file, split, label")
    rows = [row for row in rows if row["split"].strip().lower() == split]
    if not rows:
        raise ValueError(f"No rows for split {split!r} in {manifest}")
    labels = [label_value(row["label"]) for row in rows]
    paths = []
    for row in rows:
        rel = Path(row["file"].replace("\\", "/"))
        options = [rel] if rel.is_absolute() else [
            root / rel, manifest.parent / rel,
            manifest.parent / "images" / row["split"] / rel.name]
        path = next((p for p in options if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Image not found: {row['file']}")
        paths.append(path)
    dev = device()
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(ck["state_dict"], strict=True)
    net.eval().to(dev)
    probabilities = []
    print(f"Running {checkpoint.name}: {len(paths)} images on {dev}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([
                image_tensor(read_rgb(path), ck["image_size"])
                for path in paths[start:start + batch_size]]).to(dev)
            raw = torch.softmax(net(batch), dim=1)[:, 1].cpu().numpy().astype(float)
            raw = np.clip(raw, 1e-7, 1 - 1e-7)
            logits = intercept + slope * np.log(raw / (1 - raw))
            if not np.isfinite(logits).all():
                raise ValueError("Model produced nonfinite calibrated logits")
            # Stable sigmoid without overflow for large magnitude logits.
            exp_neg_abs = np.exp(-np.abs(logits))
            calibrated = np.where(logits >= 0, 1 / (1 + exp_neg_abs),
                                  exp_neg_abs / (1 + exp_neg_abs))
            probabilities.extend(calibrated.tolist())
            print(f"  Evaluated {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    protocol = ck.get("evaluation", "Training separation unverified; active classifier was fitted on all supplied images")
    return labels, probabilities, threshold, protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="WoundWatch root (default: current working directory)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Default: models/classifier_candidate.pt")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Default: data/raw/surgwound/infection_risk.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--saved-predictions", action="store_true",
                        help="Recompute candidate saved test scores without loading a model")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON output; never overwrites")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.saved_predictions and (
            args.checkpoint is not None or args.manifest is not None or args.split != "test"):
        parser.error("--saved-predictions only evaluates the candidate's saved test CSV")
    root = args.repo.resolve()
    def in_repo(path):
        return path if path.is_absolute() else root / path
    checkpoint = in_repo(args.checkpoint or Path("models/classifier_candidate.pt"))
    manifest = in_repo(args.manifest or Path("data/raw/surgwound/infection_risk.csv"))
    try:
        if args.output and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        if args.saved_predictions:
            labels, probabilities, threshold = saved_predictions(root)
            source = "Saved candidate test predictions (no new inference)"
            protocol = "Candidate: supplied test split; tuning on train+val; no patient IDs"
        else:
            labels, probabilities, threshold, protocol = infer(
                root, checkpoint, manifest, args.split, args.batch_size)
            source = f"Fresh classifier inference: {checkpoint}; split={args.split}"
        report = calculate_metrics(labels, probabilities, threshold)
        report.update(source=source, protocol=protocol,
                      scope="Classifier only; no quality-gate or defer exclusions")
        print("\n" + source)
        print("Protocol: " + protocol)
        print("Classifier only; all selected images included; positive = Elevated")
        print(f"Images: {report['n']} | Saved threshold: {threshold:.8f}")
        for key in ("accuracy", "precision", "recall", "f1"):
            value = report[key + "_percent"]
            score = "N/A (undefined denominator)" if value is None else f"{value:.2f}%"
            print(f"{key.capitalize():10s}: {score}")
        matrix = report["confusion_matrix"]
        print("\nConfusion matrix (counts; rows=actual, columns=predicted)")
        print("                  Low  Elevated")
        print(f"Actual Low      {matrix[0][0]:5d} {matrix[0][1]:9d}")
        print(f"Actual Elevated {matrix[1][0]:5d} {matrix[1][1]:9d}")
        print("Image-level evaluation; patient independence is not established.")
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
            print(f"Saved: {args.output}")
        return report
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()

