"""Write service/web/data/model_performance.json from results/ and experiments/ (single source for the Model page)."""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ..config import CFG
from .finetune_cv import OUT

LABELS = {"frozen_lr": "ResNet-18 frozen + logistic (baseline)", "frozen_r18_tuned": "ResNet-18 frozen, tuned C",
          "frozen_r50": "ResNet-50 frozen", "frozen_cnxt": "ConvNeXt-Tiny frozen", "frozen_effb0": "EfficientNet-B0 frozen",
          "frozen_dense121": "DenseNet-121 frozen", "frozen_r18_r50": "ResNet-18+50 concatenated",
          "frozen_all": "4 backbones concatenated", "ens_all": "5-backbone ensemble", "ens_big": "11-backbone ensemble",
          "ft_head_wce": "ResNet-18 head fine-tune", "ft_l4_wce": "ResNet-18 layer4 fine-tune (weighted)",
          "ft_l4_ce": "ResNet-18 layer4 fine-tune (unweighted)", "ft_l4_wce_light": "layer4 + lighting augmentation",
          "ft_l34_wce": "ResNet-18 layer3+4 fine-tune"}


def main(selected):
    ev = json.loads((CFG.results / "classifier_v2_eval.json").read_text())
    t, b = ev["test"], ev["baseline_test"]
    rows = []
    for f in sorted(OUT.glob("cv_*.json")):
        d = json.loads(f.read_text()); ps = d["per_seed"]
        m = lambda k: float(np.mean([s["crossfit_f1"][k] for s in ps]))
        rows.append(dict(name=LABELS.get(d["config"], d["config"]), key=d["config"], ap=d["ap_mean"],
                         auroc=float(np.mean([s["auroc"] for s in ps])), precision=m("precision"), recall=m("recall"), f1=m("f1"),
                         selected=d["config"] == selected))
    rows.sort(key=lambda r: -r["ap"])
    base = pd.read_csv(CFG.results / "classifier_candidate_test_predictions.csv")
    yb, pb = (base.binary_label == "Elevated").to_numpy().astype(int), base.calibrated_probability.to_numpy()
    pr, rc, _ = precision_recall_curve(yb, pb)
    curve = lambda label, r, p, prev: dict(label=label, prevalence=prev, points=[[float(x), float(y)] for x, y in zip(r[::max(1, len(r) // 80)], p[::max(1, len(p) // 80)])])
    out = dict(
        caveat="Image-level evaluation without patient IDs, 20 Elevated test images. Screening support for clinician review, not a diagnosis; no prospective clinical validation.",
        active=dict(name=f"Wound Trace v2 ({LABELS.get(selected, selected)}); baseline shown in PR chart", test=dict(
            {k: t[k] for k in ("n", "positives", "prevalence", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "auroc", "ap", "always_low_accuracy")},
            auroc_ci=t["auroc_ci"], confusion={k: t[k] for k in ("tn", "fp", "fn", "tp")})),
        baseline=dict(accuracy=b["accuracy"], precision=b["precision"], recall=b["recall"], f1=b["f1"], auroc=b["auroc"], ap=b["ap"]),
        pr_curves=[curve("v2", [p[0] for p in t["pr_curve"]], [p[1] for p in t["pr_curve"]], t["prevalence"]), curve("Baseline", rc, pr, t["prevalence"])],
        experiments=dict(dev_n=rows and 549, rows=rows,
                         note="5-fold CV x 3 seeds on the 549 train+validation images; precision/recall/F1 at cross-fitted F1-optimal thresholds. Selection rule fixed in advance: highest average precision."),
        limits=["Only 686 images carry infection labels (113 Elevated); other datasets lack them.", "No patient IDs: related photos may sit in different folds, so scores may be optimistic.",
                "The test split was seen by earlier benchmark work; it was not used to fit, select or threshold this model.", "Grad-CAM and colour maps explain model attention and measured redness, not infection.",
                "Segmenter was trained on foot ulcers; many surgical photos are flagged for retake."])
    dest = CFG.root / "service" / "web" / "data" / "model_performance.json"
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(out, indent=1)); print("wrote", dest)


if __name__ == "__main__":
    main(sys.argv[1])
