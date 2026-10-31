"""Segmenter candidate trained on every local mask-labelled set that is safe to use (FUSeg train + Medetec-224 train).

    python -m woundwatch.experiments.segmenter_v2 --epochs 10

Never overwrites models/segmenter.pt. Selection: FUSeg validation Dice. Also reports how often old/new
segmenters find a wound in SurgWound images: a coverage proxy only, since no surgical masks exist.
AZH patches are NOT used: FUSeg is derived from that data, so using it would leak into validation.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..benchmark_classifier import resolve_paths
from ..common import device, read_rgb, seed_all
from ..config import CFG
from ..inference import load_segmenter, segment_wound
from ..train_segmentation import Pairs, loss_fn, metrics
import pandas as pd

OUT = CFG.root / "experiments" / "segmenter-v2"


class Aug(Dataset):
    """Colour and zoom jitter on top of Pairs, to be less tied to foot-ulcer photo style."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        x = x * random.uniform(.8, 1.25) + random.uniform(-.3, .3)
        x = x + torch.tensor([random.uniform(-.15, .15) for _ in range(3)]).view(3, 1, 1)
        if random.random() < .5:  # zoom into a random crop, as in close-up surgical photos
            s = random.uniform(.55, .95); h, w = y.shape; ch, cw = int(h * s), int(w * s)
            t, l = random.randint(0, h - ch), random.randint(0, w - cw)
            x = F.interpolate(x[None, :, t:t + ch, l:l + cw], size=(h, w), mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y[None, None, t:t + ch, l:l + cw].float(), size=(h, w), mode="nearest")[0, 0].long()
        return x, y


def coverage(net, paths, dev):
    fr = []
    for p in paths:
        m = segment_wound(read_rgb(p), net); fr.append(float(m.mean()))
    fr = np.array(fr); return dict(found=float((fr >= CFG.minimum_wound_area_fraction).mean()), median_area=float(np.median(fr)),
                                   very_large=float((fr > .6).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); seed_all(CFG.seed); dev = device()
    net, ck = load_segmenter(); base = copy.deepcopy(net).eval()
    med = CFG.raw / "FUSeg" / "Medetec_foot_ulcer_224"
    tr = ConcatDataset([Pairs(CFG.fuseg / "train/images", CFG.fuseg / "train/labels", True), Pairs(med / "train/images", med / "train/labels", True)])
    va = Pairs(CFG.fuseg / "validation/images", CFG.fuseg / "validation/labels")
    ex = Pairs(CFG.fuseg_external / "images", CFG.fuseg_external / "labels")
    loader = DataLoader(Aug(tr), batch_size=a.batch_size, shuffle=True, num_workers=0)
    vl, el = DataLoader(va, batch_size=1), DataLoader(ex, batch_size=1)
    df = pd.read_csv(CFG.manifest); paths = [p for p, s in zip(resolve_paths(df), df.split) if s in ("train", "val")][:150]
    net.to(dev); b_val, b_ext = metrics(net, vl, dev)[0], metrics(net, el, dev)[0]
    print(f"baseline val Dice={b_val:.4f} ext Dice={b_ext:.4f} train={len(tr)}", flush=True)
    enc = [p for n, p in net.named_parameters() if n.startswith(("conv1", "bn1", "layer"))]
    dec = [p for n, p in net.named_parameters() if not n.startswith(("conv1", "bn1", "layer"))]
    opt = torch.optim.AdamW([dict(params=enc, lr=2e-5), dict(params=dec, lr=2e-4)], weight_decay=1e-4)
    best, best_d, best_ep, hist = copy.deepcopy(net.state_dict()), b_val, 0, []
    for ep in range(1, a.epochs + 1):
        net.train(); tot = 0
        for x, y in loader:
            opt.zero_grad(set_to_none=True); loss = loss_fn(net(x.to(dev)), y.to(dev)); loss.backward(); opt.step(); tot += float(loss) * len(y)
        d, i = metrics(net, vl, dev); hist.append(dict(epoch=ep, loss=tot / len(tr), val_dice=d, val_iou=i)); print(hist[-1], flush=True)
        if d > best_d: best_d, best, best_ep = d, copy.deepcopy(net.state_dict()), ep
    net.load_state_dict(best); e_d, e_i = metrics(net, el, dev)
    net.eval(); base.to(dev)
    rep = dict(train_images=len(tr), baseline_val_dice=b_val, candidate_val_dice=best_d, selected_epoch=best_ep, baseline_external_dice=b_ext, candidate_external_dice=e_d,
               surgwound_coverage_baseline=coverage(base, paths, dev), surgwound_coverage_candidate=coverage(net, paths, dev),
               note="Medetec-224 train and test share a source, so the external figure is in-distribution. Coverage is a proxy: no surgical masks exist.", history=hist)
    ck.update(state_dict={k: v.cpu() for k, v in best.items()}, retraining=rep); torch.save(ck, OUT / "candidate.pt")
    (OUT / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps({k: v for k, v in rep.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
