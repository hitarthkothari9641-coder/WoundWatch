"""Train DeepLabV3+ with ResNet-18 encoder; validate and externally test."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import models
from .common import binary_mask,device,find_mask,read_rgb,seed_all
from .config import CFG

class CBR(nn.Sequential):
    def __init__(self,ci,co,k=3,d=1):
        super().__init__(nn.Conv2d(ci,co,k,padding=(k//2)*d,dilation=d,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True))

class ASPP(nn.Module):
    def __init__(self):
        super().__init__(); self.branches=nn.ModuleList([CBR(512,128,1),CBR(512,128,3,6),CBR(512,128,3,12),CBR(512,128,3,18)])
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(512,128,1),nn.ReLU(inplace=True))
        self.project=CBR(640,128,1); self.drop=nn.Dropout2d(.1)
    def forward(self,x):
        z=F.interpolate(self.pool(x),size=x.shape[-2:],mode="bilinear",align_corners=False)
        return self.drop(self.project(torch.cat([*(b(x) for b in self.branches),z],1)))

class DeepLabV3Plus18(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__(); weights=None
        if pretrained:
            try: weights=models.ResNet18_Weights.DEFAULT
            except Exception: weights=None
        try: enc=models.resnet18(weights=weights)
        except Exception as e:
            print(f"WARNING: pretrained segmentation encoder unavailable ({type(e).__name__}); using random weights")
            enc=models.resnet18(weights=None); weights=None
        self.pretrained_weights_used=weights is not None
        self.conv1,self.bn1,self.relu,self.maxpool=enc.conv1,enc.bn1,enc.relu,enc.maxpool
        self.layer1,self.layer2,self.layer3,self.layer4=enc.layer1,enc.layer2,enc.layer3,enc.layer4
        self.aspp=ASPP(); self.low=CBR(64,48,1); self.decoder=nn.Sequential(CBR(176,128),CBR(128,128),nn.Dropout2d(.1)); self.head=nn.Conv2d(128,2,1)
    def forward(self,x):
        size=x.shape[-2:]; z=self.maxpool(self.relu(self.bn1(self.conv1(x)))); low=self.layer1(z)
        z=self.layer4(self.layer3(self.layer2(low))); z=F.interpolate(self.aspp(z),size=low.shape[-2:],mode="bilinear",align_corners=False)
        return F.interpolate(self.head(self.decoder(torch.cat([z,self.low(low)],1))),size=size,mode="bilinear",align_corners=False)

class Pairs(Dataset):
    def __init__(self,idir,ldir,augment=False):
        self.size=CFG.segment_size; self.augment=augment
        files=sorted(p for p in Path(idir).glob("*") if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg"})
        self.pairs=[(p,find_mask(p,ldir)) for p in files]
        if not self.pairs: raise ValueError(f"No matching image-mask pairs in {idir} / {ldir}")
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ip,mp=self.pairs[i]; im=read_rgb(ip)
        with Image.open(mp) as m: m.load(); mask=binary_mask(m)
        mask=Image.fromarray(mask.astype(np.uint8)*255)
        if self.augment:
            if random.random()<.5: im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT); mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random()<.5: im=im.transpose(Image.Transpose.FLIP_TOP_BOTTOM); mask=mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            angle=random.uniform(-15,15); im=im.rotate(angle,resample=Image.Resampling.BILINEAR); mask=mask.rotate(angle,resample=Image.Resampling.NEAREST)
        h,w=self.size; im=im.resize((w,h),getattr(self,'resize_resampling',Image.Resampling.BILINEAR)); mask=mask.resize((w,h),Image.Resampling.NEAREST)
        x=np.asarray(im,dtype=np.float32)/255
        x=(x-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy((np.asarray(mask)>0).astype(np.int64))

def loss_fn(logits,y):
    ce=F.cross_entropy(logits,y); p=torch.softmax(logits,1)[:,1]; t=(y==1).float()
    inter=(p*t).sum((1,2)); dice=(2*inter+1)/(p.sum((1,2))+t.sum((1,2))+1)
    return ce+1-dice.mean()

@torch.inference_mode()
def metrics(model,loader,dev):
    model.eval(); inter=union=ps=ts=0
    for x,y in loader:
        p=model(x.to(dev)).argmax(1).cpu().numpy().astype(bool); t=y.numpy().astype(bool)
        inter+=np.logical_and(p,t).sum(); union+=np.logical_or(p,t).sum(); ps+=p.sum(); ts+=t.sum()
    return float(2*inter/(ps+ts)) if ps+ts else 1.,float(inter/union) if union else 1.

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--epochs",type=int,default=CFG.segmentation_epochs); ap.add_argument("--batch-size",type=int,default=None); ap.add_argument("--scratch",action="store_true"); a=ap.parse_args()
    if a.epochs<1: ap.error("--epochs must be >=1")
    CFG.create_dirs(); seed_all(CFG.seed); dev=device(); bs=a.batch_size or (CFG.segmentation_gpu_batch if dev.type=="cuda" else CFG.segmentation_cpu_batch)
    root=CFG.fuseg
    tr=Pairs(root/"train/images",root/"train/labels",True); va=Pairs(root/"validation/images",root/"validation/labels")
    ex=Pairs(CFG.fuseg_external/"images",CFG.fuseg_external/"labels")
    kwargs={"num_workers":0,"pin_memory":dev.type=="cuda"}; train=DataLoader(tr,batch_size=bs,shuffle=True,**kwargs)
    vl=DataLoader(va,batch_size=1,**kwargs); el=DataLoader(ex,batch_size=1,**kwargs)
    net=DeepLabV3Plus18(not a.scratch).to(dev); print(f"Device={dev}, pretrained encoder={net.pretrained_weights_used}, train/val/external={len(tr)}/{len(va)}/{len(ex)}")
    opt=torch.optim.AdamW(net.parameters(),lr=CFG.learning_rate)
    for ep in range(a.epochs):
        net.train(); total=0
        for x,y in train:
            x,y=x.to(dev,non_blocking=dev.type=="cuda"),y.to(dev,non_blocking=dev.type=="cuda"); opt.zero_grad(set_to_none=True); loss=loss_fn(net(x),y); loss.backward(); opt.step(); total+=float(loss.detach())*len(y)
        d,i=metrics(net,vl,dev); print(f"epoch {ep+1}/{a.epochs}: CE+Dice loss={total/len(tr):.4f}; validation Dice={d:.4f}, IoU={i:.4f}")
    vd,vi=metrics(net,vl,dev); ed,ei=metrics(net,el,dev)
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in net.state_dict().items()},"architecture":"deeplabv3plus_resnet18","image_size":list(CFG.segment_size),"pretrained_encoder_used":net.pretrained_weights_used,"validation_dice":vd,"validation_iou":vi,"external_dice":ed,"external_iou":ei,"seed":CFG.seed},CFG.models/"segmenter.pt")
    report={"validation":{"dice":vd,"iou":vi,"n":len(va)},"external_medetec":{"dice":ed,"iou":ei,"n":len(ex)}}
    (CFG.results/"segmentation_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_examples(net,va,dev,CFG.results/"seg_examples",12)
    print(f"Validation Dice={vd:.4f}, IoU={vi:.4f}; external Medetec Dice={ed:.4f}, IoU={ei:.4f}.")
    print("Dice=2*overlap/(predicted+true pixels); IoU=overlap/union. Saved models/segmenter.pt")

def save_examples(net,ds,dev,dest,n):
    dest.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(CFG.seed); ids=rng.choice(len(ds),min(n,len(ds)),replace=False)
    mean=np.array([.485,.456,.406]); std=np.array([.229,.224,.225]); net.eval()
    with torch.inference_mode():
        for k,ix in enumerate(ids,1):
            x,y=ds[int(ix)]; pred=net(x[None].to(dev)).argmax(1)[0].cpu().numpy().astype(bool)
            rgb=np.clip((x.numpy().transpose(1,2,0)*std+mean)*255,0,255).astype(np.uint8)
            rgba=np.zeros((*pred.shape,4),np.uint8); rgba[pred]=[255,40,40,100]
            base=Image.fromarray(rgb).convert("RGBA"); base=Image.alpha_composite(base,Image.fromarray(rgba,"RGBA"))
            base.convert("RGB").save(dest/f"seg_{k:02d}.jpg",quality=92)

if __name__=="__main__": main()
