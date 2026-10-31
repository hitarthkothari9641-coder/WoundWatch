"""Auxiliary Kaggle wound-type pretraining and SurgWound transfer benchmark.

Kaggle folder labels describe wound morphology/type, not infection. This script
uses them only for a 10-class auxiliary pretraining task, then fine-tunes two
ResNet-18 initialization strategies on SurgWound train+val and compares their
image-level OOF predictions. It never uses the SurgWound test split.

The Kaggle archive has paired ``mirrored_*`` files. Only the non-mirrored
original is used for auxiliary training. Patient IDs are unavailable in
SurgWound, so these results are not patient-independent estimates.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .common import device, image_tensor, read_rgb, seed_all
from .config import CFG

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GEOMETRIC_AUGMENT = transforms.RandomAffine(
    degrees=15, translate=(0.04, 0.04), scale=(0.96, 1.04))


class PathImages(Dataset):
    def __init__(self, paths, labels=None, augment=False):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_rgb(self.paths[index])
        if self.augment:
            image = transforms.RandomHorizontalFlip(p=0.5)(image)
            image = GEOMETRIC_AUGMENT(image)
        tensor = image_tensor(image, CFG.classifier_size)
        if self.labels is None:
            return tensor, index
        return tensor, int(self.labels[index])


def kaggle_originals(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle wound dataset folder missing: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(f"Expected wound-type subfolders under {root}")
    classes = [p.name for p in class_dirs]
    paths, labels, pair_rows, counts = [], [], [], {}
    for label, folder in enumerate(class_dirs):
        all_images = sorted(p for p in folder.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        mirrors = {p.name[len("mirrored_"):].casefold(): p for p in all_images
                   if p.name.casefold().startswith("mirrored_")}
        base_images = [p for p in all_images
                       if not p.name.casefold().startswith("mirrored_")]
        base_by_name = {p.name.casefold(): p for p in base_images}
        if len(mirrors) != len(all_images) - len(base_images):
            raise ValueError(f"Duplicate mirrored filenames found in Kaggle class {folder.name}")
        for mirror_name, mirror_path in mirrors.items():
            original = base_by_name.get(mirror_name)
            if original is None:
                raise ValueError(f"Mirrored Kaggle image has no matching original: {mirror_path.name}")
            pair_rows.append((folder.name, original, mirror_path))
        if len(base_images) != len(mirrors):
            raise ValueError(f"Expected one mirrored copy per original in class {folder.name}; "
                             f"found {len(base_images)} originals and {len(mirrors)} mirrors")
        counts[folder.name] = len(base_images)
        paths.extend(base_images)
        labels.extend([label] * len(base_images))
    if len(pair_rows) != sum(counts.values()):
        raise ValueError("Kaggle mirror-pair audit did not match the original-image count")
    return paths, np.asarray(labels, dtype=np.int64), classes, counts, pair_rows


def image_before_after_report(pair_rows, output, limit=8):
    # Show the dataset's mirrored duplicate construction; the mirror is excluded
    # from pretraining and no image pixels are altered during this audit step.
    chosen = []
    for class_name in sorted({r[0] for r in pair_rows}):
        chosen.append(next(r for r in pair_rows if r[0] == class_name))
        if len(chosen) >= limit:
            break
    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = np.asarray([axes])
    for row_index, (axrow, (name, original, mirror)) in enumerate(zip(axes, chosen)):
        for ax, path in zip(axrow, (original, mirror)):
            ax.imshow(read_rgb(path)); ax.axis("off")
        axrow[0].set_ylabel(name, rotation=0, labelpad=55, va="center", fontsize=10)
    axes[0, 0].set_title("Original used for pretraining", fontsize=10)
    axes[0, 1].set_title("Mirrored duplicate excluded", fontsize=10)
    fig.suptitle("Kaggle mirror-pair audit", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def augmentation_preview(paths, classes, output):
    selected = []
    for class_id, name in enumerate(classes):
        match = next((p for p in paths if p.parent.name == name), None)
        if match is not None:
            selected.append((name, match))
    selected = selected[:8]
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3.1 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row, (name, path) in zip(axes, selected):
        image = read_rgb(path)
        seed_all(CFG.seed + len(name))
        augmented = transforms.RandomHorizontalFlip(p=1.0)(image)
        augmented = transforms.RandomAffine(degrees=(12, 12), translate=(0.03, 0.02),
                                             scale=(1.0, 1.0))(augmented)
        row[0].imshow(image)
        row[1].imshow(augmented)
        row[0].set_ylabel(name, rotation=0, labelpad=52, va="center", fontsize=10)
        row[0].axis("off"); row[1].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 1].set_title("Geometric transform", fontsize=10)
    fig.suptitle("Training augmentation preview (geometry only; color unchanged)",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_model(initialization, source_state=None, num_classes=2):
    if initialization == "imagenet":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif initialization == "kaggle":
        if source_state is None:
            raise ValueError("Kaggle initialization needs the wound-type checkpoint")
        net = models.resnet18(weights=None)
        trunk = {k: v for k, v in source_state.items() if not k.startswith("fc.")}
        missing, unexpected = net.load_state_dict(trunk, strict=False)
        if unexpected or any(not k.startswith("fc.") for k in missing):
            raise RuntimeError("Kaggle pretrained trunk did not match ResNet-18")
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train_model(net, paths, labels, dev, epochs, batch, tag, seed):
    seed_all(seed)
    dataset = PathImages(paths, labels, augment=True)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0,
                        pin_memory=dev.type == "cuda")
    counts = np.bincount(labels, minlength=net.fc.out_features)
    if np.any(counts == 0):
        raise ValueError(f"Training split lacks a class for {tag}")
    class_weights = len(labels) / (len(counts) * counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(
        class_weights, dtype=torch.float32, device=dev))
    net.to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=CFG.learning_rate,
                                 weight_decay=1e-4)
    for epoch in range(epochs):
        net.train(); loss_total = 0.0
        for x, y in loader:
            x = x.to(dev, non_blocking=dev.type == "cuda")
            y = y.to(dev, non_blocking=dev.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                enabled=dev.type == "cuda"):
                loss = loss_fn(net(x), y)
            loss.backward(); optimizer.step()
            loss_total += float(loss.detach()) * len(y)
        print(f"{tag} epoch {epoch+1}/{epochs}: weighted CE={loss_total/len(dataset):.4f}")
    net.eval()
    return net


@torch.inference_mode()
def predict_probability(net, paths, dev, batch):
    loader = DataLoader(PathImages(paths), batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=dev.type == "cuda")
    result = np.empty(len(paths), dtype=np.float64)
    for x, indices in loader:
        logits = net(x.to(dev, non_blocking=dev.type == "cuda"))
        result[indices.numpy()] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
    return result


def gradcam_rgb(net, image, dev):
    """Return a normalized Grad-CAM overlay; this is not a causal explanation."""
    net.eval(); layer = net.layer4[-1].conv2; state = {}
    def capture(_module, _inputs, output):
        state["activation"] = output
        output.register_hook(lambda grad: state.__setitem__("gradient", grad))
    handle = layer.register_forward_hook(capture)
    x = image_tensor(image, CFG.classifier_size).unsqueeze(0).to(dev)
    net.zero_grad(set_to_none=True)
    with torch.enable_grad():
        score = net(x)[0, 1]
        score.backward()
    handle.remove()
    activation, gradient = state["activation"].detach(), state["gradient"].detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear",
                        align_corners=False)[0, 0].cpu().numpy()
    cam -= cam.min(); cam /= max(float(cam.max()), 1e-8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3].astype(np.float32)
    return np.clip(0.52 * rgb + 0.48 * heat, 0, 1)


def save_gradient_comparison(image_path, low_model, kaggle_model, dev, label, path):
    image = read_rgb(image_path)
    before = gradcam_rgb(low_model, image, dev)
    after = gradcam_rgb(kaggle_model, image, dev)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(image); axes[0].set_title(f"Input (SurgWound {label})")
    axes[1].imshow(before); axes[1].set_title("ImageNet-init Grad-CAM")
    axes[2].imshow(after); axes[2].set_title("Kaggle-type-pretrained Grad-CAM")
    for ax in axes: ax.axis("off")
    fig.suptitle("Grad-CAM sensitivity comparison (not an explanation)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibrate_oof(y, raw, folds):
    calibrated = np.empty_like(raw)
    for fold in np.unique(folds):
        train, valid = folds != fold, folds == fold
        cal = LogisticRegression(C=1e6, max_iter=2000, random_state=CFG.seed)
        clipped = np.clip(raw[train], 1e-7, 1 - 1e-7)
        cal.fit(np.log(clipped / (1 - clipped)).reshape(-1, 1), y[train])
        p = np.clip(raw[valid], 1e-7, 1 - 1e-7)
        calibrated[valid] = cal.predict_proba(
            np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    p = np.clip(raw, 1e-7, 1 - 1e-7)
    final = LogisticRegression(C=1e6, max_iter=2000, random_state=CFG.seed)
    final.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return calibrated, final


def summarize(y, raw, calibrated):
    fpr, tpr, thresholds = roc_curve(y, calibrated)
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    pred = calibrated >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "elevated_n": int(y.sum()),
        "elevated_prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, raw)),
        "auprc": float(average_precision_score(y, raw)),
        "calibrated_brier": float(brier_score_loss(y, calibrated)),
        "youden_threshold_calibrated": threshold,
        "youden_sensitivity": float(tp / max(1, tp + fn)),
        "youden_specificity": float(tn / max(1, tn + fp)),
        "youden_accuracy": float(accuracy_score(y, pred)),
        "youden_confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def save_final_checkpoint(initialization, source_state, paths, y, dev, epochs,
                          batch, calibrator, metrics, out):
    net = make_model(initialization, source_state)
    net = train_model(net, paths, y, dev, epochs, batch,
                      f"final {initialization}", CFG.seed + 9000)
    raw = np.clip(np.asarray(metrics["chosen_oof_probabilities"]), 1e-7, 1 - 1e-7)
    fpr, tpr, thresholds = roc_curve(y, metrics["chosen_oof_calibrated"])
    youden = float(thresholds[int(np.argmax(tpr - fpr))])
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "backbone": "resnet18", "head": "fine_tuned_binary",
        "initialization": initialization,
        "pretraining_task": "10-class Kaggle wound-type classification" if initialization == "kaggle" else "ImageNet",
        "image_size": list(CFG.classifier_size), "class_order": ["Low", "Elevated"],
        "imagenet_normalization": True, "pretrained_weights_used": True,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "threshold_youden": youden,
        "threshold_sensitivity_090": None,
        "defer_half_width": CFG.defer_half_width,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in {"chosen_oof_probabilities", "chosen_oof_calibrated"}},
        "patient_independent": False, "seed": CFG.seed,
    }
    torch.save(payload, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-gradcam", action="store_true")
    args = parser.parse_args()
    if min(args.pretrain_epochs, args.finetune_epochs) < 1 or args.folds < 2:
        parser.error("epochs must be >=1 and folds >=2")

    CFG.create_dirs(); seed_all(CFG.seed); dev = device()
    kaggle_paths, kaggle_y, classes, class_counts, mirror_pairs = kaggle_originals(CFG.kaggle)
    print("Kaggle auxiliary labels are wound types, never infection labels.")
    print(f"Kaggle original images={len(kaggle_paths)}; mirrored duplicates excluded="
          f"{len(mirror_pairs)}; wound-type classes={len(classes)}")
    for name, count in class_counts.items(): print(f"  {name}: {count} originals")
    image_before_after_report(mirror_pairs, CFG.results / "kaggle_mirror_audit.png")
    augmentation_preview(kaggle_paths, classes, CFG.results / "augmentation_before_after.png")

    batch = args.batch_size or (16 if dev.type == "cuda" else 8)
    pretrain = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrain.fc = nn.Linear(pretrain.fc.in_features, len(classes))
    pretrain = train_model(pretrain, kaggle_paths, kaggle_y, dev,
                           args.pretrain_epochs, batch, "Kaggle wound-type pretrain",
                           CFG.seed + 100)
    source_state = {k: v.detach().cpu() for k, v in pretrain.state_dict().items()}
    source_path = CFG.models / "kaggle_woundtype_resnet18.pt"
    torch.save({"state_dict": source_state, "classes": classes,
                "class_counts_originals": class_counts,
                "mirror_pairs_excluded": len(mirror_pairs),
                "task": "wound type; not infection", "seed": CFG.seed}, source_path)

    manifest = pd.read_csv(CFG.manifest)
    dev_df = manifest[manifest.split.astype(str).str.lower().isin(
        ["train", "val", "validation"])].reset_index(drop=True)
    if not {"file", "split", "label"}.issubset(manifest.columns):
        raise ValueError("Manifest must contain file, split, label")
    y = (dev_df.label.to_numpy() != "Low").astype(np.int64)
    if np.bincount(y, minlength=2).min() < args.folds:
        raise ValueError("Insufficient Low/Elevated samples for requested folds")
    surg_paths = []
    for row in dev_df.itertuples(index=False):
        rel = Path(str(row.file))
        path = next((p for p in (rel, CFG.surgwound / rel,
                                  CFG.surgwound / "images" / str(row.split) / rel.name)
                     if p.is_file()), None)
        if path is None: raise FileNotFoundError(f"SurgWound image missing: {rel}")
        surg_paths.append(path)
    folds = StratifiedKFold(args.folds, shuffle=True, random_state=CFG.seed)
    oof = {"imagenet": np.full(len(y), np.nan), "kaggle": np.full(len(y), np.nan)}
    fold_ids = np.zeros(len(y), dtype=np.int64)
    gradcam_dir = CFG.results / "transfer_gradcam"

    for fold, (tr, va) in enumerate(folds.split(np.zeros(len(y)), y), start=1):
        fold_ids[va] = fold
        fold_models = {}
        for init in ("imagenet", "kaggle"):
            seed_all(CFG.seed + 1000 * fold)
            net = make_model(init, source_state if init == "kaggle" else None)
            net = train_model(net, [surg_paths[i] for i in tr], y[tr], dev,
                              args.finetune_epochs, batch,
                              f"fold {fold} {init}", CFG.seed + 1000 * fold)
            oof[init][va] = predict_probability(net, [surg_paths[i] for i in va], dev, batch)
            fold_models[init] = net
        if not args.skip_gradcam:
            chosen = []
            for label_value in (0, 1):
                matching = va[y[va] == label_value]
                if len(matching): chosen.append(int(matching[0]))
            for local_case, index in enumerate(chosen, start=1):
                label_name = "Elevated" if y[index] else "Low"
                save_gradient_comparison(
                    surg_paths[index], fold_models["imagenet"], fold_models["kaggle"],
                    dev, label_name,
                    gradcam_dir / f"fold_{fold:02d}_{local_case}_{label_name}.png")
        del fold_models
        if dev.type == "cuda": torch.cuda.empty_cache()
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("Incomplete OOF predictions")

    analysis = {
        "protocol": "Same stratified image-level folds on supplied SurgWound train+val only; "
                    "no patient IDs; supplied test split not evaluated by this script",
        "kaggle_auxiliary_pretraining": {
            "task": "10-class wound type; not infection",
            "source_image_count": int(len(kaggle_paths)),
            "mirrored_duplicate_images_excluded": int(len(mirror_pairs)),
            "class_counts_after_mirror_exclusion": class_counts,
            "class_names": classes,
            "pretrain_epochs": args.pretrain_epochs,
        },
        "surgwound_development": {"n": int(len(y)), "elevated_n": int(y.sum()),
                                  "finetune_epochs": args.finetune_epochs,
                                  "folds": int(args.folds)},
    }
    calibrated, calibrators = {}, {}
    for init in ("imagenet", "kaggle"):
        calibrated[init], calibrators[init] = calibrate_oof(y, oof[init], fold_ids)
        analysis[init] = summarize(y, oof[init], calibrated[init])

    analysis["comparison"] = {
        "delta_kaggle_minus_imagenet_auroc": analysis["kaggle"]["auroc"] - analysis["imagenet"]["auroc"],
        "delta_kaggle_minus_imagenet_auprc": analysis["kaggle"]["auprc"] - analysis["imagenet"]["auprc"],
        "interpretation": "Exploratory model-selection comparison; image-level OOF and two candidates "
                          "were compared on these folds, so this is not a final unbiased estimate.",
    }
    analysis["limits"] = [
        "No patient IDs; image-level folds may contain related wound images.",
        "Kaggle wound-type labels are not infection outcomes and were used only for auxiliary pretraining.",
        "Kaggle mirror files were excluded by matching mirrored_ filenames to originals; source permissions remain to be verified.",
        "The supplied test split was not used in this experiment; it should remain locked for a new independent evaluation protocol.",
        "Grad-CAM maps are model sensitivity visualizations, not explanations or clinical evidence.",
    ]

    results_path = CFG.results / "kaggle_transfer_analysis.json"
    results_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    rows = []
    for i, row in dev_df.iterrows():
        rows.append({"file": str(row.file), "split": str(row.split),
                     "label": str(row.label), "fold": int(fold_ids[i]),
                     "imagenet_oof_probability": float(oof["imagenet"][i]),
                     "kaggle_transfer_oof_probability": float(oof["kaggle"][i]),
                     "imagenet_calibrated_oof": float(calibrated["imagenet"][i]),
                     "kaggle_transfer_calibrated_oof": float(calibrated["kaggle"][i])})
    pd.DataFrame(rows).to_csv(CFG.results / "kaggle_transfer_oof.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["ImageNet init", "Kaggle wound-type\npretraining"]
    for ax, metric, title in zip(axes, ("auroc", "auprc"), ("AUROC", "AUPRC")):
        vals = [analysis[k][metric] for k in ("imagenet", "kaggle")]
        ax.bar(names, vals, color=["#5577aa", "#d17a3c"])
        ax.set_ylim(0, 1); ax.set_ylabel(title); ax.set_title(f"Development OOF {title}")
        for idx, value in enumerate(vals): ax.text(idx, value + .02, f"{value:.3f}", ha="center")
    fig.suptitle("Kaggle auxiliary pretraining: SurgWound development comparison")
    fig.tight_layout(); fig.savefig(CFG.results / "kaggle_transfer_comparison.png", dpi=160)
    plt.close(fig)

    winner = max(("imagenet", "kaggle"), key=lambda name: analysis[name]["auroc"])
    winner_oof = oof[winner]
    checkpoint = save_final_checkpoint(
        winner, source_state if winner == "kaggle" else None,
        surg_paths, y, dev, args.finetune_epochs, batch, calibrators[winner],
        {**analysis, "chosen_oof_probabilities": winner_oof,
         "chosen_oof_calibrated": calibrated[winner]},
        CFG.models / "classifier_transfer_candidate.pt")
    analysis["selected_by_development_oof_auroc"] = winner
    # Write the selected model after recording the model-selection decision.
    results_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("\nSurgWound train+val OOF results (exploratory; image-level):")
    for name in ("imagenet", "kaggle"):
        score = analysis[name]
        print(f"{name}: AUROC={score['auroc']:.4f}; AUPRC={score['auprc']:.4f} "
              f"(Elevated prevalence={score['elevated_prevalence']:.4f}); "
              f"Brier={score['calibrated_brier']:.4f}; Youden accuracy={score['youden_accuracy']:.4f}")
    print(f"Selected development candidate: {winner}; saved {checkpoint}")
    print(f"Reports: {results_path}, {CFG.results/'kaggle_transfer_comparison.png'}, "
          f"{CFG.results/'augmentation_before_after.png'}, {CFG.results/'kaggle_mirror_audit.png'}, "
          f"{gradcam_dir}")
    print("Accuracy depends on a threshold; AUROC/AUPRC measure ranking. None of these "
          "image-level results establish performance on unseen patients or clinical readiness.")


if __name__ == "__main__":
    main()
