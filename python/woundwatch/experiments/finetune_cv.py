"""Development-only CV comparison for the Elevated-vs-Low classifier (MODEL_IMPROVEMENT_TODO P0-P2).

Predeclared selection objective: pooled out-of-fold average precision (AP), ties broken
by Elevated F1 then balanced accuracy. The supplied test split is never read here.
Early stopping uses an inner stratified hold-out of each outer-training fold.
Thresholds for reported operating points are cross-fitted: the threshold applied to
fold k is chosen on the OOF scores of the other folds.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torchvision import models
from torchvision.transforms import v2

from ..benchmark_classifier import resolve_paths
from ..common import image_tensor, read_rgb
from ..config import CFG
from ..ensemble import build_backbone

OUT = CFG.root / "experiments" / "classifier"
CONFIGS = {
    # name: (unfreeze, loss weighting, augmentation, head lr, backbone lr)
    "frozen_lr": dict(kind="logreg", backbones=["resnet18"], C=0.001),  # baseline (benchmark_classifier setting)
    "frozen_r18_tuned": dict(kind="logreg", backbones=["resnet18"], C=None),
    "frozen_r50": dict(kind="logreg", backbones=["resnet50"], C=None),
    "frozen_cnxt": dict(kind="logreg", backbones=["convnext_tiny"], C=None),
    "frozen_effb0": dict(kind="logreg", backbones=["efficientnet_b0"], C=None),
    "frozen_dense121": dict(kind="logreg", backbones=["densenet121"], C=None),
    "frozen_r18_r50": dict(kind="logreg", backbones=["resnet18", "resnet50"], C=None),
    "ens_all": dict(kind="logreg", ens=True, backbones=["resnet18", "resnet50", "convnext_tiny", "efficientnet_b0", "densenet121"], C=None),
    "ens_big": dict(kind="logreg", ens=True, backbones=["resnet18", "resnet50", "resnet101", "convnext_tiny", "convnext_small", "efficientnet_b0", "efficientnet_b3", "densenet121", "swin_t", "vit_b_16", "regnet_y_8gf"], C=None),
    "frozen_all": dict(kind="logreg", backbones=["resnet18", "resnet50", "convnext_tiny", "efficientnet_b0"], C=None),
    "ft_l4_wce": dict(kind="ft", unfreeze="layer4", weighted=True, aug="geo", head_lr=3e-4, bb_lr=3e-5),
    "ft_l4_ce": dict(kind="ft", unfreeze="layer4", weighted=False, aug="geo", head_lr=3e-4, bb_lr=3e-5),
    "ft_l4_wce_light": dict(kind="ft", unfreeze="layer4", weighted=True, aug="geo+light", head_lr=3e-4, bb_lr=3e-5),
    "ft_l34_wce": dict(kind="ft", unfreeze="layer3", weighted=True, aug="geo", head_lr=3e-4, bb_lr=1e-5),
    "ft_head_wce": dict(kind="ft", unfreeze="none", weighted=True, aug="geo", head_lr=1e-3, bb_lr=0.0),
}


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_tensors():
    df = pd.read_csv(CFG.manifest)
    paths = resolve_paths(df)
    sig = hashlib.sha256("|".join(f"{p}:{p.stat().st_size}" for p in paths).encode()).hexdigest()[:12]
    cache = OUT / f"tensors_{sig}.npy"
    OUT.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        x = np.load(cache)
    else:
        x = np.stack([image_tensor(read_rgb(p), CFG.classifier_size).numpy() for p in paths]).astype(np.float16)
        np.save(cache, x)
    y = (df.label != "Low").to_numpy().astype(int)
    return df, torch.from_numpy(x), y, sig


def make_model(cfg):
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)  # raises if weights unavailable; never silently random
    net.fc = nn.Linear(net.fc.in_features, 2)
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith("fc") or (cfg["unfreeze"] == "layer4" and n.startswith("layer4")) or (
            cfg["unfreeze"] == "layer3" and n.startswith(("layer3", "layer4")))
    return net


def set_train_mode(net, cfg):
    net.train()
    for name, m in net.named_modules():  # frozen layers keep ImageNet BN statistics
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def augmenter(kind):
    ops = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
           v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1))]
    if kind == "geo+light":
        ops.append(v2.ColorJitter(brightness=0.1, contrast=0.1))  # mild; hue/saturation untouched (redness)
    return v2.Compose(ops)


@torch.no_grad()
def predict(net, x, dev, bs=64):
    net.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(torch.softmax(net(x[i:i + bs].float().to(dev)), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_ft(x, y, tr, va, cfg, seed, dev, max_epochs=30, patience=5, fixed_epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_model(cfg).to(dev)
    head = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("fc")]
    bb = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("fc")]
    groups = [{"params": head, "lr": cfg["head_lr"]}] + ([{"params": bb, "lr": cfg["bb_lr"]}] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    counts = np.bincount(y[tr], minlength=2)
    w = torch.tensor(len(tr) / (2 * counts), dtype=torch.float32, device=dev) if cfg["weighted"] else None
    lossf = nn.CrossEntropyLoss(weight=w)
    aug = augmenter(cfg["aug"])
    best, best_state, best_ep, bad = -1, None, 0, 0
    epochs = fixed_epochs or max_epochs
    for ep in range(1, epochs + 1):
        set_train_mode(net, cfg)
        order = np.random.permutation(tr)
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            if len(idx) < 2: continue
            xb = torch.stack([aug(x[j].float()) for j in idx]).to(dev)
            yb = torch.from_numpy(y[idx]).to(dev)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        if va is not None:
            ap = average_precision_score(y[va], predict(net, x[va], dev))
            if ap > best:
                best, best_ep, bad = ap, ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state is not None: net.load_state_dict(best_state)
    return net, best_ep


_FEATS = {}
def backbone_features(name, x, dev):
    """Frozen ImageNet pooled features, mean of original and h-flipped view; computed once per backbone."""
    if name in _FEATS: return _FEATS[name]
    net = build_backbone(name)
    net = net.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            b = x[i:i + 32].float().to(dev)
            out.append(((net(b) + net(torch.flip(b, [3]))) / 2).flatten(1).cpu())
    _FEATS[name] = torch.cat(out).numpy()
    return _FEATS[name]


def choose_C(f, y, tr):
    """Pick LR strength by inner 3-fold AP on the training part only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    inner = StratifiedKFold(3, shuffle=True, random_state=0)
    def inner_ap(c):
        sc = []
        for a, b in inner.split(tr, y[tr]):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000)).fit(f[tr[a]], y[tr[a]])
            sc.append(average_precision_score(y[tr[b]], m.predict_proba(f[tr[b]])[:, 1]))
        return np.mean(sc)
    return max([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], key=inner_ap)


def frozen_lr_scores(x, y, tr, te, dev, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if cfg.get("ens"):  # average of per-backbone LR probabilities, each with its own inner-CV C
        return np.mean([frozen_lr_scores(x, y, tr, te, dev, {**cfg, "ens": False, "backbones": [b]}) for b in cfg["backbones"]], 0)
    f = np.concatenate([backbone_features(b, x, dev) for b in cfg["backbones"]], 1)
    mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    C = cfg["C"] if cfg["C"] is not None else choose_C(f, y, tr)
    return mk(C).fit(f[tr], y[tr]).predict_proba(f[te])[:, 1]


def best_f1_threshold(y, p):
    ts = np.unique(p)
    f = [2 * ((p >= t) & (y == 1)).sum() / max(1, (p >= t).sum() + (y == 1).sum()) for t in ts]
    return float(ts[int(np.argmax(f))])


def op_metrics(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return dict(threshold=float(t), accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                f1=2 * prec * rec / max(1e-12, prec + rec), balanced_accuracy=(rec + spec) / 2,
                specificity=spec, tn=tn, fp=fp, fn=fn, tp=tp)


def summarize(y, p, folds):
    cross_t = np.zeros(len(y))
    for k in np.unique(folds):  # threshold for fold k from the other folds only
        cross_t[folds == k] = best_f1_threshold(y[folds != k], p[folds != k])
    pred = p >= cross_t
    tp, fp = ((pred) & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    prec, rec, spec = tp / max(1, tp + fp), tp / max(1, tp + fn), tn / max(1, tn + fp)
    return dict(ap=float(average_precision_score(y, p)), auroc=float(roc_auc_score(y, p)),
                brier_uncalibrated=float(brier_score_loss(y, p)), prevalence=float(y.mean()),
                crossfit_f1=dict(accuracy=float((tp + tn) / len(y)), precision=float(prec), recall=float(rec),
                                 f1=float(2 * prec * rec / max(1e-12, prec + rec)),
                                 balanced_accuracy=float((rec + spec) / 2)),
                pooled_f1_threshold=best_f1_threshold(y, p))


def run(name, seeds, folds=5):
    cfg = CONFIGS[name]
    df, x, y, sig = load_tensors()
    dev_idx = np.where(df.split.isin(["train", "val"]))[0]  # test split excluded
    dev = pick_device()
    per_seed, oofs, best_eps = [], [], []
    for seed in seeds:
        t0 = time.time()
        oof, fold_id = np.zeros(len(dev_idx)), np.zeros(len(dev_idx), int)
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for k, (tr_l, te_l) in enumerate(skf.split(dev_idx, y[dev_idx])):
            tr, te = dev_idx[tr_l], dev_idx[te_l]
            if cfg["kind"] == "logreg":
                oof[te_l] = frozen_lr_scores(x[dev_idx], y[dev_idx], tr_l, te_l, dev, cfg)
            else:
                tr_in, va_in = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
                net, ep = train_ft(x, y, tr_in, va_in, cfg, seed, dev)
                best_eps.append(ep)
                oof[te_l] = predict(net, x[te], dev)
            fold_id[te_l] = k
        m = summarize(y[dev_idx], oof, fold_id)
        m["seed"], m["seconds"] = seed, round(time.time() - t0, 1)
        per_seed.append(m); oofs.append(oof)
        print(f"{name} seed {seed}: AP {m['ap']:.4f} AUROC {m['auroc']:.4f} F1 {m['crossfit_f1']['f1']:.4f} "
              f"P {m['crossfit_f1']['precision']:.3f} R {m['crossfit_f1']['recall']:.3f} ({m['seconds']}s)", flush=True)
    ens = summarize(y[dev_idx], np.mean(oofs, 0), fold_id)  # fold_id of last seed only used for thresholds
    res = dict(config=name, cfg=cfg, seeds=list(seeds), dev_n=len(dev_idx), data_sig=sig,
               per_seed=per_seed, ap_mean=float(np.mean([s["ap"] for s in per_seed])),
               ap_sd=float(np.std([s["ap"] for s in per_seed])), seed_mean_oof=ens,
               median_best_epoch=int(np.median(best_eps)) if best_eps else None)
    (OUT / f"cv_{name}.json").write_text(json.dumps(res, indent=2))
    np.save(OUT / f"oof_{name}.npy", np.mean(oofs, 0))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    for c in a.configs: run(c, a.seeds)
