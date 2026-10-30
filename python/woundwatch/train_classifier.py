"""Train the Low vs Elevated classifier; exports OOF metrics and final checkpoint."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from .common import device, read_rgb, seed_all
from .config import CFG

class ImageSet(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels = paths, labels
        ops = []
        if augment:
            # Geometric transforms only; colour is retained as a potential signal.
            ops = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                   transforms.RandomAffine(degrees=15, translate=(.04,.04))]
        self.tf = transforms.Compose(ops+[transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(read_rgb(self.paths[i])), torch.tensor(int(self.labels[i]), dtype=torch.long)

def resolve_paths(df):
    out=[]
    for r in df.itertuples(index=False):
        rel=Path(str(r.file)); opts=[rel, CFG.surgwound/rel, CFG.surgwound/"images"/str(r.split)/rel.name]
        p=next((x for x in opts if x.is_file()),None)
        if p is None: raise FileNotFoundError(f"Manifest image not found: {rel}")
        out.append(p)
    return out

def cache_images(paths):
    root=CFG.processed/"classifier_224"; root.mkdir(parents=True,exist_ok=True); result=[]
    h,w=CFG.classifier_size
    for i,p in enumerate(paths):
        st=p.stat(); q=root/f"{i:05d}_{st.st_size}_{st.st_mtime_ns}.png"
        if not q.exists(): read_rgb(p).resize((w,h),Image.Resampling.BICUBIC).save(q,format="PNG")
        result.append(q)
    print(f"Resized image cache ready: {len(result)} images")
    return result

def make_model(pretrained):
    try:
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        model=models.resnet18(weights=weights); used=pretrained
    except Exception as e:
        print(f"WARNING: pretrained weights unavailable ({type(e).__name__}); training from scratch")
        model=models.resnet18(weights=None); used=False
    model.fc=nn.Linear(model.fc.in_features,2)
    return model,used

def train(model,loader,weights,epochs,dev,tag):
    model.to(dev); opt=torch.optim.AdamW(model.parameters(),lr=CFG.learning_rate)
    loss_fn=nn.CrossEntropyLoss(weight=torch.as_tensor(weights,dtype=torch.float32,device=dev))
    for ep in range(epochs):
        model.train(); loss_sum=0
        for x,y in loader:
            x,y=x.to(dev,non_blocking=dev.type=="cuda"),y.to(dev,non_blocking=dev.type=="cuda"); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=="cuda"):
                loss=loss_fn(model(x),y)
            loss.backward(); opt.step(); loss_sum+=float(loss.detach())*len(y)
        print(f"{tag} epoch {ep+1}/{epochs}: weighted crossentropy={loss_sum/len(loader.dataset):.4f}")
    return model

@torch.inference_mode()
def predict(model,loader,dev):
    model.eval(); ans=[]
    for x,_ in loader: ans.extend(torch.softmax(model(x.to(dev,non_blocking=dev.type=="cuda")),1)[:,1].cpu().numpy())
    return np.asarray(ans,float)

def logit(p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7); return np.log(p/(1-p)).reshape(-1,1)

def stats(y,p,t):
    tn,fp,fn,tp=confusion_matrix(y,np.asarray(p)>=t,labels=[0,1]).ravel()
    return tp/(tp+fn or 1),tn/(tn+fp or 1),(int(tn),int(fp),int(fn),int(tp))

def threshold_for_sensitivity(y,p,target=.9):
    vals=np.r_[0,np.unique(p),1.00001]
    candidates=[(stats(y,p,t)[1],t) for t in vals if stats(y,p,t)[0]>=target]
    return float(max((s,t) for s,t in candidates)[1]) if candidates else 0.0

def threshold_for_specificity(y,p,target=.9):
    vals=np.r_[0,np.unique(p),1.00001]
    candidates=[(stats(y,p,t)[0],t) for t in vals if stats(y,p,t)[1]>=target]
    return float(max((sens,t) for sens,t in candidates)[1]) if candidates else 1.00001

def bootstrap_ci(y,p,seed,n=CFG.bootstrap_replicates):
    rng=np.random.default_rng(seed); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); auc=[]
    for _ in range(n):
        ix=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
        auc.append(roc_auc_score(y[ix],p[ix]))
    return np.percentile(auc,[2.5,97.5]).tolist()

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--epochs",type=int,default=CFG.classifier_epochs)
    ap.add_argument("--folds",type=int,default=CFG.folds); ap.add_argument("--batch-size",type=int,default=None)
    ap.add_argument("--scratch",action="store_true"); a=ap.parse_args()
    if a.epochs<1 or a.folds<2: ap.error("epochs >= 1 and folds >= 2 required")
    CFG.create_dirs(); seed_all(CFG.seed); dev=device(); batch=a.batch_size or (CFG.classifier_gpu_batch if dev.type=="cuda" else CFG.classifier_cpu_batch)
    if not CFG.manifest.is_file(): raise FileNotFoundError(f"Missing manifest {CFG.manifest}")
    df=pd.read_csv(CFG.manifest)
    if not {"file","split","label"}.issubset(df.columns): raise ValueError("Manifest needs file, split, label columns")
    if not set(df.label.unique())<={"Low","Medium","High"}: raise ValueError("Labels must be Low, Medium, High")
    y=(df.label.to_numpy()!="Low").astype(np.int64)
    if np.bincount(y,minlength=2).min()<a.folds: raise ValueError("Not enough samples in a class for requested folds")
    orig=resolve_paths(df); paths=cache_images(orig)
    print(f"Device={dev}; backbone=resnet18; n={len(y)} (Low={(y==0).sum()}, Elevated={(y==1).sum()}); image-stratified {a.folds}-fold CV (no patient IDs)")
    oof=np.full(len(y),np.nan); fold_ids=np.zeros(len(y),int); pretrained_used=None
    cv=StratifiedKFold(a.folds,shuffle=True,random_state=CFG.seed)
    for fold,(tr,va) in enumerate(cv.split(np.zeros(len(y)),y),1):
        seed_all(CFG.seed+fold); fold_ids[va]=fold
        model,used=make_model(not a.scratch)
        if pretrained_used is None:
            pretrained_used=used
            print("Initialization: pretrained ImageNet weights" if used else "Initialization: random weights (from scratch)")
        train_ds=ImageSet([paths[i] for i in tr],y[tr],True); val_ds=ImageSet([paths[i] for i in va],y[va])
        loader=DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers=0,pin_memory=dev.type=="cuda")
        vloader=DataLoader(val_ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=dev.type=="cuda")
        counts=np.bincount(y[tr],minlength=2); model=train(model,loader,len(tr)/(2*counts),a.epochs,dev,f"fold {fold}")
        oof[va]=predict(model,vloader,dev); del model,loader,vloader
        if dev.type=="cuda": torch.cuda.empty_cache()
    if not np.isfinite(oof).all(): raise RuntimeError("OOF predictions incomplete")
    crosscal=np.empty_like(oof)
    for fold in range(1,a.folds+1):
        cal=LogisticRegression(C=1e6,random_state=CFG.seed).fit(logit(oof[fold_ids!=fold]),y[fold_ids!=fold])
        crosscal[fold_ids==fold]=cal.predict_proba(logit(oof[fold_ids==fold]))[:,1]
    cal=LogisticRegression(C=1e6,random_state=CFG.seed).fit(logit(oof),y)
    ty=threshold_for_sensitivity(y,crosscal,0.9); fpr,tpr,thr=roc_curve(y,crosscal); youden=float(thr[np.argmax(tpr-fpr)])
    spec90thr=threshold_for_specificity(y,oof,.90); auc=roc_auc_score(y,oof); apval=average_precision_score(y,oof); ci=bootstrap_ci(y,oof,CFG.seed+1)
    print("\nOOF pooled metrics: AUROC ranks positive vs negative (0.5 chance); AUPRC uses elevated prevalence as baseline.")
    print(f"AUROC {auc:.4f} (bootstrap 95% CI {ci[0]:.4f}-{ci[1]:.4f}); AUPRC {apval:.4f}; prevalence {y.mean():.4f}")
    print(f"Sensitivity at >=90% specificity: {stats(y,oof,spec90thr)[0]:.4f} (specificity={stats(y,oof,spec90thr)[1]:.4f}; raw threshold {spec90thr:.5f})")
    print(f"Cross-fitted Platt Brier={brier_score_loss(y,crosscal):.5f} (mean squared probability error; lower is better)")
    print(f"Calibrated Youden threshold={youden:.5f}, sensitivity/specificity/confusion={stats(y,crosscal,youden)}")
    print(f"Calibrated sensitivity>=0.90 threshold={ty:.5f}, sensitivity/specificity/confusion={stats(y,crosscal,ty)}")
    pd.DataFrame({"file":[str(p.relative_to(CFG.root)) for p in orig],"split":df.split,"label":df.label,
                  "binary_label":np.where(y,"Elevated","Low"),"fold":fold_ids,"oof_probability":oof,
                  "cross_fitted_calibrated_probability":crosscal}).to_csv(CFG.results/"cv_results.csv",index=False)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ix=np.argsort(crosscal); bins=np.array_split(ix,min(10,len(ix)))
    fig,ax=plt.subplots(); ax.plot([0,1],[0,1],"k--"); ax.plot([crosscal[b].mean() for b in bins],[y[b].mean() for b in bins],"o-")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Predicted elevated probability",ylabel="Observed elevated fraction",title="Cross-fitted calibration"); ax.grid(); fig.savefig(CFG.results/"calibration_plot.png",dpi=160); plt.close(fig)
    seed_all(CFG.seed+999); model,final_pretrained=make_model(not a.scratch)
    ds=ImageSet(paths,y,True); loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0,pin_memory=dev.type=="cuda")
    model=train(model,loader,len(y)/(2*np.bincount(y,minlength=2)),a.epochs,dev,"final")
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"backbone":"resnet18",
        "image_size":list(CFG.classifier_size),"class_order":["Low","Elevated"],"imagenet_normalization":True,
        "pretrained_weights_used":bool(final_pretrained),"platt_intercept":float(cal.intercept_[0]),
        "platt_slope":float(cal.coef_[0,0]),"threshold_youden":youden,"threshold_sensitivity_090":ty,
        "threshold_specificity_090_raw":spec90thr,
        "defer_half_width":CFG.defer_half_width,"metrics":{"auroc":float(auc),"auroc_ci95":ci,"auprc":float(apval),"prevalence":float(y.mean()),"brier":float(brier_score_loss(y,crosscal))},"seed":CFG.seed},CFG.models/"classifier.pt")
    print(f"Saved classifier: {CFG.models/'classifier.pt'} and OOF results: {CFG.results/'cv_results.csv'}")

if __name__=="__main__": main()
