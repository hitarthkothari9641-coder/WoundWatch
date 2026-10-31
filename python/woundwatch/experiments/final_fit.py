"""Fit the selected frozen-backbone ensemble on ALL development images, then score the locked test split once.

    python -m woundwatch.experiments.final_fit ens_all --out models/classifier_v2.pt

Platt calibration and the decision threshold come from the development out-of-fold scores written by
finetune_cv (seed-mean OOF); the test split is read only for the single final evaluation, through the
same EnsembleClassifier + inference path the service uses.
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CFG
from ..ensemble import build_ensemble
from .finetune_cv import (CONFIGS, OUT, backbone_features, best_f1_threshold, choose_C, load_tensors, op_metrics,
                          pick_device)


def bootstrap_ci(y, p, fn, n=2000, seed=0):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=[k for k, v in CONFIGS.items() if v["kind"] == "logreg" and v.get("ens")])
    ap.add_argument("--out", default=str(CFG.models / "classifier_v2.pt"))
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    df, x, y, sig = load_tensors()
    dev_idx = np.where(df.split.isin(["train", "val"]))[0]
    test_idx = np.where(df.split == "test")[0]
    dev = pick_device()

    members = []
    for name in cfg["backbones"]:
        f = backbone_features(name, x, dev)
        C = choose_C(f[dev_idx], y[dev_idx], np.arange(len(dev_idx)))
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(f[dev_idx], y[dev_idx])
        sc, lr = pipe[0], pipe[1]
        members.append(dict(name=name, C=float(C), mean=torch.from_numpy(sc.mean_.astype(np.float32)), scale=torch.from_numpy(sc.scale_.astype(np.float32)),
                            coef=torch.from_numpy(lr.coef_[0].astype(np.float32)), intercept=float(lr.intercept_[0])))  # tensors: checkpoint must load with weights_only=True
        print(f"{name}: C={C}", flush=True)

    # Platt + threshold from development OOF only.
    oof = np.load(OUT / f"oof_{a.config}.npy"); yd = y[dev_idx]
    z = np.log(np.clip(oof, 1e-7, 1 - 1e-7) / (1 - np.clip(oof, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(z, yd)
    slope, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    cal = platt.predict_proba(z)[:, 1]
    threshold = best_f1_threshold(yd, cal)

    ck = dict(kind="ensemble_lr", members=members, backbone="ensemble:" + "+".join(cfg["backbones"]), image_size=(224, 224),
              platt_intercept=intercept, platt_slope=slope, threshold_youden=float(threshold),
              defer_half_width=CFG.defer_half_width, pretrained_weights_used=True, selection="max pooled dev OOF AP; F1 threshold",
              config=a.config, data_sig=sig, dev_n=int(len(dev_idx)))
    torch.save(ck, a.out)

    # Single locked-test evaluation through the deployed module.
    net = build_ensemble(ck).to(dev)
    with torch.no_grad():
        raw = torch.cat([torch.softmax(net(x[test_idx[i:i + 32]].float().to(dev)), 1)[:, 1].cpu()
                         for i in range(0, len(test_idx), 32)]).numpy()
    zt = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / (1 - np.clip(raw, 1e-7, 1 - 1e-7)))
    pt = 1 / (1 + np.exp(-(intercept + slope * zt)))
    yt = y[test_idx]
    m = op_metrics(yt, pt, threshold)
    prec, rec, _ = precision_recall_curve(yt, pt)
    base = pd.read_csv(CFG.results / "classifier_candidate_test_predictions.csv")
    yb, pb = (base.binary_label == "Elevated").to_numpy().astype(int), base.calibrated_probability.to_numpy()
    bm = op_metrics(yb, pb, 0.2785421515235395)  # development-selected Youden threshold of the baseline candidate
    res = dict(config=a.config, members=[{k: v for k, v in mm.items() if k in ("name", "C")} for mm in members],
               threshold=float(threshold), platt=dict(intercept=intercept, slope=slope),
               test=dict(n=int(len(yt)), positives=int(yt.sum()), prevalence=float(yt.mean()), **m,
                         ap=float(average_precision_score(yt, pt)), auroc=float(roc_auc_score(yt, pt)),
                         auroc_ci=bootstrap_ci(yt, pt, roc_auc_score), ap_ci=bootstrap_ci(yt, pt, average_precision_score),
                         always_low_accuracy=float(1 - yt.mean()), pr_curve=[[float(r), float(p)] for r, p in zip(rec, prec)]),
               baseline_test=dict(n=int(len(yb)), **bm, ap=float(average_precision_score(yb, pb)), auroc=float(roc_auc_score(yb, pb))),
               note="Test split already seen by earlier benchmark work; it was not used to fit, select, calibrate or threshold this model.")
    (CFG.results / "classifier_v2_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res["test"][k] for k in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "ap", "auroc", "tn", "fp", "fn", "tp")}, indent=1))
    print("baseline:", {k: round(bm[k], 4) for k in ("accuracy", "precision", "recall", "f1")})


if __name__ == "__main__":
    main()
