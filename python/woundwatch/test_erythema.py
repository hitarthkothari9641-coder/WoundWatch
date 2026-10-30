"""Explore whether image-level erythema separates Low and Elevated labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .common import read_rgb
from .config import CFG
from .color_features import feature_row
from .inference import load_segmenter,segment_wound

def main():
    tab=pd.read_csv(CFG.manifest); net,_=load_segmenter(); labels=[]; values=[]; skipped=0
    for r in tab.itertuples(index=False):
        p=CFG.surgwound/str(r.file)
        if not p.is_file(): p=CFG.surgwound/"images"/str(r.split)/p.name
        im=read_rgb(p); m=segment_wound(im,net)
        try: f=feature_row(im,m,CFG.ring_dilation_pixels)
        except ValueError: skipped+=1; continue
        labels.append(int(r.label!="Low")); values.append(f["erythema_a_mean_difference"])
    y=np.asarray(labels); x=np.asarray(values)
    if not len(y) or len(np.unique(y))<2: raise ValueError("Not enough valid segmented images from both classes to calculate erythema AUROC")
    auc=roc_auc_score(y,x); rng=np.random.default_rng(CFG.seed); pos=np.flatnonzero(y); neg=np.flatnonzero(1-y); boot=[]
    for _ in range(2000):
        ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]; boot.append(roc_auc_score(y[ix],x[ix]))
    ci=np.percentile(boot,[2.5,97.5]); print(f"Erythema a* AUROC={auc:.4f} (image-level bootstrap 95% CI {ci[0]:.4f}-{ci[1]:.4f}); 0.5 is chance; valid={len(y)}, skipped={skipped}.")

if __name__=="__main__": main()
