"""Leakage-aware model-selection benchmark for the supplied SurgWound split.

Uses nested CV only within the supplied train/val partitions to select a
regularized linear head over frozen ImageNet ResNet-18 features. The supplied
test partition is scored once at the end and is never used for tuning.
Because patient identifiers are absent, this is not a patient-independent
evaluation; exact/near-duplicate screening cannot reconstruct patient IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from .common import device, image_tensor, read_rgb, seed_all
from .config import CFG


class FeatureImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_rgb(self.paths[index])
        return image_tensor(image, CFG.classifier_size), index


def resolve_paths(df):
    paths = []
    for row in df.itertuples(index=False):
        rel = Path(str(row.file))
        options = (rel, CFG.surgwound / rel,
                   CFG.surgwound / "images" / str(row.split) / rel.name)
        path = next((candidate for candidate in options if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Manifest image not found: {rel}")
        paths.append(path)
    return paths


@torch.inference_mode()
def extract_features(paths, dev, batch_size):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    encoder = nn.Sequential(*list(model.children())[:-1]).eval().to(dev)
    signature_items = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                       for p in paths]
    signature = hashlib.sha256(json.dumps(
        {"weights": str(weights), "size": CFG.classifier_size,
         "files": signature_items}, sort_keys=True).encode()).hexdigest()
    cache = CFG.processed / "resnet18_imagenet_features.npz"
    if cache.is_file():
        try:
            saved = np.load(cache, allow_pickle=False)
            if str(saved["signature"].item()) == signature:
                feats = saved["features"].astype(np.float32, copy=False)
                if feats.shape == (len(paths), 512):
                    print(f"Using cached frozen-backbone features for {len(paths)} images")
                    return feats
        except (OSError, ValueError, KeyError):
            pass

    loader = DataLoader(FeatureImages(paths), batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=dev.type == "cuda")
    features = np.empty((len(paths), 512), dtype=np.float32)
    for x, indices in loader:
        output = encoder(x.to(dev, non_blocking=dev.type == "cuda")).flatten(1)
        features[indices.numpy()] = output.float().cpu().numpy()
    np.savez_compressed(cache, features=features, signature=np.asarray(signature))
    print(f"Extracted and cached frozen-backbone features for {len(paths)} images")
    return features


def candidate_pipeline(c_value, class_weight):
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            C=float(c_value), class_weight=class_weight,
            solver="lbfgs", max_iter=2000, random_state=CFG.seed))])


def select_model(x, y, folds, seed):
    if np.bincount(y, minlength=2).min() < folds:
        raise ValueError(f"Need at least {folds} examples from each class for CV")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    candidates = [(c, weight) for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
                  for weight in (None, "balanced")]
    scored = []
    for c_value, class_weight in candidates:
        aucs = []
        for tr, va in cv.split(x, y):
            estimator = candidate_pipeline(c_value, class_weight)
            estimator.fit(x[tr], y[tr])
            aucs.append(roc_auc_score(y[va], estimator.predict_proba(x[va])[:, 1]))
        scored.append((float(np.mean(aucs)), -float(c_value),
                       class_weight is None, c_value, class_weight))
    best = max(scored)
    return {"mean_cv_auroc": best[0], "C": float(best[3]),
            "class_weight": best[4]}


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def threshold_at_sensitivity(y, p, target=0.90):
    thresholds = np.r_[0.0, np.unique(p), 1.00001]
    valid = []
    for threshold in thresholds:
        pred = p >= threshold
        tp = np.sum((y == 1) & pred)
        fn = np.sum((y == 1) & ~pred)
        tn = np.sum((y == 0) & ~pred)
        fp = np.sum((y == 0) & pred)
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        if sensitivity >= target:
            valid.append((specificity, float(threshold)))
    return max(valid)[1] if valid else 0.0


def threshold_at_specificity(y, p, target=0.90):
    thresholds = np.r_[0.0, np.unique(p), 1.00001]
    valid = []
    for threshold in thresholds:
        pred = p >= threshold
        tp = np.sum((y == 1) & pred)
        fn = np.sum((y == 1) & ~pred)
        tn = np.sum((y == 0) & ~pred)
        fp = np.sum((y == 0) & pred)
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        if specificity >= target:
            valid.append((sensitivity, float(threshold)))
    return max(valid)[1] if valid else 1.00001


def stratified_auc_ci(y, p, seed, replicates):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(replicates):
        sample = np.r_[rng.choice(pos, len(pos), replace=True),
                       rng.choice(neg, len(neg), replace=True)]
        values.append(roc_auc_score(y[sample], p[sample]))
    return np.percentile(values, [2.5, 97.5]).tolist()


def fit_checkpoint(pipe, dev, thresholds, calibration, metrics, seed):
    pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrained.fc = nn.Linear(512, 2)
    scaler, logistic = pipe.named_steps["scale"], pipe.named_steps["logistic"]
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    effective_weights = logistic.coef_[0] / scale
    effective_bias = logistic.intercept_[0] - np.dot(
        logistic.coef_[0], scaler.mean_ / scale)
    with torch.no_grad():
        pretrained.fc.weight.zero_()
        pretrained.fc.bias.zero_()
        pretrained.fc.weight[1].copy_(torch.as_tensor(effective_weights,
                                                       dtype=torch.float32))
        pretrained.fc.bias[1] = float(effective_bias)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in pretrained.state_dict().items()},
        "backbone": "resnet18", "head": "frozen_imagenet_logistic",
        "image_size": list(CFG.classifier_size),
        "class_order": ["Low", "Elevated"], "imagenet_normalization": True,
        "pretrained_weights_used": True,
        "platt_intercept": float(calibration.intercept_[0]),
        "platt_slope": float(calibration.coef_[0, 0]),
        "threshold_youden": float(thresholds["youden"]),
        "threshold_sensitivity_090": float(thresholds["sensitivity_090"]),
        "threshold_specificity_090": float(thresholds["specificity_090"]),
        "defer_half_width": CFG.defer_half_width,
        "metrics": metrics, "seed": seed,
        "evaluation": "supplied test split; tuning only on supplied train+val",
        "patient_independent": False,
    }
    out = CFG.models / "classifier_candidate.pt"
    torch.save(payload, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int,
                        default=CFG.bootstrap_replicates)
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.bootstrap_replicates < 100:
        parser.error("fold counts must be >=2 and bootstrap replicates >=100")

    CFG.create_dirs()
    seed_all(CFG.seed)
    dev = device()
    manifest = pd.read_csv(CFG.manifest)
    if not {"file", "split", "label"}.issubset(manifest.columns):
        raise ValueError("SurgWound manifest must have file, split, label columns")
    if not set(manifest["label"].dropna().unique()) <= {"Low", "Medium", "High"}:
        raise ValueError("Unexpected SurgWound label; expected Low, Medium, High")
    development = manifest[manifest["split"].astype(str).str.lower().isin(
        ["train", "val", "validation"])].reset_index(drop=True)
    test = manifest[manifest["split"].astype(str).str.lower().eq("test")].reset_index(drop=True)
    if development.empty or test.empty:
        raise ValueError("Expected nonempty train/val and test partitions in manifest")
    y_dev = (development.label.to_numpy() != "Low").astype(np.int64)
    y_test = (test.label.to_numpy() != "Low").astype(np.int64)
    folds = min(args.outer_folds, int(np.bincount(y_dev, minlength=2).min()))
    if folds < 2:
        raise ValueError("Not enough examples in both development classes for CV")

    dev_paths, test_paths = resolve_paths(development), resolve_paths(test)
    all_paths = dev_paths + test_paths
    batch = args.batch_size or (32 if dev.type == "cuda" else 16)
    print(f"Frozen ImageNet ResNet-18 features on {dev}; development={len(dev_paths)} "
          f"(Elevated={int(y_dev.sum())}), held-out test={len(test_paths)} "
          f"(Elevated={int(y_test.sum())}). Patient IDs unavailable.")
    features = extract_features(all_paths, dev, batch)
    x_dev, x_test = features[:len(dev_paths)], features[len(dev_paths):]

    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CFG.seed)
    oof_raw = np.full(len(y_dev), np.nan)
    fold_ids = np.zeros(len(y_dev), dtype=np.int64)
    selected = []
    for fold, (tr, va) in enumerate(outer.split(x_dev, y_dev), start=1):
        tuning = select_model(x_dev[tr], y_dev[tr],
                              min(args.inner_folds, int(np.bincount(y_dev[tr], minlength=2).min())),
                              CFG.seed + fold)
        estimator = candidate_pipeline(tuning["C"], tuning["class_weight"])
        estimator.fit(x_dev[tr], y_dev[tr])
        oof_raw[va] = estimator.predict_proba(x_dev[va])[:, 1]
        fold_ids[va] = fold
        selected.append(tuning)
        print(f"Nested fold {fold}/{folds}: selected C={tuning['C']:g}, "
              f"class_weight={tuning['class_weight']}, inner AUROC={tuning['mean_cv_auroc']:.3f}")
    if not np.isfinite(oof_raw).all():
        raise RuntimeError("Nested out-of-fold predictions are incomplete")

    oof_calibrated = np.empty_like(oof_raw)
    for fold in range(1, folds + 1):
        train_mask, valid_mask = fold_ids != fold, fold_ids == fold
        cal_fold = LogisticRegression(C=1e6, random_state=CFG.seed, max_iter=2000)
        cal_fold.fit(logit(oof_raw[train_mask]), y_dev[train_mask])
        oof_calibrated[valid_mask] = cal_fold.predict_proba(logit(oof_raw[valid_mask]))[:, 1]
    calibrator = LogisticRegression(C=1e6, random_state=CFG.seed, max_iter=2000)
    calibrator.fit(logit(oof_raw), y_dev)

    final_tune = select_model(x_dev, y_dev,
                              min(5, int(np.bincount(y_dev, minlength=2).min())),
                              CFG.seed + 1000)
    final_model = candidate_pipeline(final_tune["C"], final_tune["class_weight"])
    final_model.fit(x_dev, y_dev)
    test_raw = final_model.predict_proba(x_test)[:, 1]
    test_calibrated = calibrator.predict_proba(logit(test_raw))[:, 1]
    fpr, tpr, thresholds = roc_curve(y_dev, oof_calibrated)
    youden_idx = int(np.argmax(tpr - fpr))
    threshold_youden = float(thresholds[youden_idx])
    threshold_sens = threshold_at_sensitivity(y_dev, oof_calibrated, 0.90)
    threshold_spec = threshold_at_specificity(y_dev, oof_calibrated, 0.90)

    auc_oof = float(roc_auc_score(y_dev, oof_raw))
    auc_test = float(roc_auc_score(y_test, test_raw))
    ci_test = stratified_auc_ci(y_test, test_raw, CFG.seed + 5,
                                args.bootstrap_replicates)
    ap_oof = float(average_precision_score(y_dev, oof_raw))
    ap_test = float(average_precision_score(y_test, test_raw))
    brier_oof = float(brier_score_loss(y_dev, oof_calibrated))
    brier_test = float(brier_score_loss(y_test, test_calibrated))

    def operating_metrics(prob, threshold):
        pred = prob >= threshold
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        return {"threshold_from_development": float(threshold),
                "sensitivity": float(tp / max(1, tp + fn)),
                "specificity": float(tn / max(1, tn + fp)),
                "accuracy": float(accuracy_score(y_test, pred)),
                "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}

    metrics = {
        "nested_development_oof": {
            "n": int(len(y_dev)), "elevated_prevalence": float(y_dev.mean()),
            "auroc": auc_oof, "auprc": ap_oof,
            "calibrated_brier": brier_oof,
        },
        "heldout_test": {
            "n": int(len(y_test)), "elevated_n": int(y_test.sum()),
            "elevated_prevalence": float(y_test.mean()),
            "auroc": auc_test, "auroc_image_bootstrap_ci95": ci_test,
            "auprc": ap_test, "auprc_prevalence_baseline": float(y_test.mean()),
            "calibrated_brier": brier_test,
            "youden_operating_point": operating_metrics(test_calibrated, threshold_youden),
            "sensitivity_090_operating_point": operating_metrics(test_calibrated, threshold_sens),
            "specificity_090_operating_point": operating_metrics(test_calibrated, threshold_spec),
        },
        "selected_model": {**final_tune, "backbone": "resnet18",
                            "head": "frozen_imagenet_logistic"},
        "cv_fold_selections": selected,
        "protocol": "Nested stratified image-level CV on supplied train+val; "
                    "single evaluation on supplied test split; no patient IDs",
    }
    checkpoint = fit_checkpoint(final_model, dev,
                                {"youden": threshold_youden,
                                 "sensitivity_090": threshold_sens,
                                 "specificity_090": threshold_spec},
                                calibrator, metrics, CFG.seed)
    pd.DataFrame({"file": test["file"], "label": test["label"],
                  "binary_label": np.where(y_test == 1, "Elevated", "Low"),
                  "raw_probability": test_raw,
                  "calibrated_probability": test_calibrated}).to_csv(
                      CFG.results / "classifier_candidate_test_predictions.csv", index=False)
    pd.DataFrame({"file": development["file"], "label": development["label"],
                  "binary_label": np.where(y_dev == 1, "Elevated", "Low"),
                  "fold": fold_ids, "nested_oof_probability": oof_raw,
                  "cross_fitted_calibrated_probability": oof_calibrated}).to_csv(
                      CFG.results / "classifier_candidate_development_oof.csv", index=False)
    summary_path = CFG.results / "classifier_candidate_benchmark.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nNested development OOF: AUROC ranks positive above negative (0.5 chance); "
          f"AUROC={auc_oof:.4f}, AUPRC={ap_oof:.4f} vs prevalence={y_dev.mean():.4f}, "
          f"calibrated Brier={brier_oof:.4f} (lower is better).")
    print("Held-out test (one evaluation): AUROC with image-bootstrap 95% CI="
          f"{auc_test:.4f} [{ci_test[0]:.4f}, {ci_test[1]:.4f}], "
          f"AUPRC={ap_test:.4f} vs prevalence={y_test.mean():.4f}, "
          f"calibrated Brier={brier_test:.4f}.")
    print("Test sensitivity/specificity/accuracy at development-selected Youden threshold: "
          f"{operating_metrics(test_calibrated, threshold_youden)}")
    print("Test metrics at threshold selected for >=90% development OOF sensitivity: "
          f"{operating_metrics(test_calibrated, threshold_sens)}")
    print("Test sensitivity at threshold selected for >=90% development OOF specificity: "
          f"{operating_metrics(test_calibrated, threshold_spec)}")
    print("Accuracy is threshold-dependent; AUROC/AUPRC measure ranking; AUPRC's "
          "reference baseline is the Elevated prevalence. Test CI resamples images, "
          "not patients, because patient IDs are absent.")
    print(f"Candidate checkpoint (not the active service model): {checkpoint}")
    print(f"Saved benchmark: {summary_path}")


if __name__ == "__main__":
    main()
