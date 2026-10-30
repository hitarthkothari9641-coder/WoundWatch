"""Load checkpoints and run quality gate, wound segmentation and risk prediction."""
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import models
from torch import nn
from .common import device,image_tensor,read_rgb
from .config import CFG
from .train_segmentation import DeepLabV3Plus18

def load_classifier(path=None):
    path=path or CFG.models/"classifier.pt"
    if not path.is_file(): raise FileNotFoundError(f"Classifier checkpoint missing: {path}")
    ck=torch.load(path,map_location="cpu",weights_only=True); net=models.resnet18(weights=None); net.fc=nn.Linear(net.fc.in_features,2); net.load_state_dict(ck["state_dict"]); net.eval().to(device()); return net,ck

def load_segmenter(path=None):
    path=path or CFG.models/"segmenter.pt"
    if not path.is_file(): raise FileNotFoundError(f"Segmenter checkpoint missing: {path}")
    ck=torch.load(path,map_location="cpu",weights_only=True); net=DeepLabV3Plus18(False); net.load_state_dict(ck["state_dict"]); net.eval().to(device()); return net,ck

@torch.inference_mode()
def segment_wound(image,net=None):
    if isinstance(image,(str,bytes,Path)): image=read_rgb(image)
    if net is None: net,_=load_segmenter()
    x=image_tensor(image,CFG.segment_size).unsqueeze(0).to(device())
    predicted=(net(x).argmax(1)[0].cpu().numpy()==1)
    return np.asarray(Image.fromarray(predicted.astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST))>0

def wound_and_ring(image,mask,dilation=12):
    m=np.asarray(mask,dtype=bool)
    if m.shape!= (image.height,image.width):
        m=np.asarray(Image.fromarray(m.astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST))>0
    size=max(3,2*int(dilation)+1); size += 1-size%2
    dilated=np.asarray(Image.fromarray(m.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(size)))>0
    return m, np.logical_and(dilated,~m)

def quality_check(image,segmenter=None,thresholds=None):
    thresholds=thresholds or {"blur_variance_minimum":CFG.blur_variance_minimum,
        "max_clipped_fraction":CFG.maximum_clipped_fraction,
        "min_wound_area_fraction":CFG.minimum_wound_area_fraction,
        "max_wound_area_fraction":CFG.maximum_wound_area_fraction}
    rgb=np.asarray(image.convert("RGB"),dtype=np.float64)/255
    gray=.2989360213*rgb[...,0]+.5870430745*rgb[...,1]+.1140209043*rgb[...,2]
    # Discrete 3x3 Laplacian, edge replicated.
    k=np.array([[0,1,0],[1,-4,1],[0,1,0]],float); p=np.pad(gray,1,mode="edge")
    lap=sum(k[i,j]*p[i:i+gray.shape[0],j:j+gray.shape[1]] for i in range(3) for j in range(3))
    blur=float(lap.var()); clipped=float(np.mean((gray<=.01)|(gray>=.99))); reasons=[]
    if blur<thresholds["blur_variance_minimum"]: reasons.append("Image may be blurred")
    if clipped>thresholds["max_clipped_fraction"]: reasons.append("Excessive dark/bright clipping")
    mask=segment_wound(image,segmenter); area=float(mask.mean())
    if area<thresholds["min_wound_area_fraction"]: reasons.append("Predicted wound area below minimum")
    if area>thresholds["max_wound_area_fraction"]: reasons.append("Predicted wound area above maximum")
    return {"passed":not reasons,"reasons":reasons,"blur_variance":blur,"clipped_fraction":clipped,"wound_area_fraction":area,"thresholds":thresholds}

@torch.inference_mode()
def predict_image(image,segmenter=None,classifier=None,ck=None):
    if segmenter is None: segmenter,_=load_segmenter()
    if classifier is None: classifier,ck=load_classifier()
    q=quality_check(image,segmenter)
    if not q["passed"]: return {"status":"retake","probability":None,"risk_band":"unknown","reasons":q["reasons"]}
    x=image_tensor(image,ck["image_size"]).unsqueeze(0).to(device()); raw=float(torch.softmax(classifier(x),1)[0,1].cpu())
    z=math.log(np.clip(raw,1e-7,1-1e-7)/(1-np.clip(raw,1e-7,1-1e-7)))
    prob=1/(1+math.exp(-float(ck["platt_intercept"]+ck["platt_slope"]*z))); t=float(ck["threshold_youden"]); width=float(ck.get("defer_half_width",CFG.defer_half_width))
    if abs(prob-t)<=width: return {"status":"defer","probability":prob,"risk_band":"defer","reasons":["Probability is within defer band around decision threshold"]}
    return {"status":"ok","probability":prob,"risk_band":"elevated" if prob>=t else "low","reasons":[]}

def predict_ssi(path):
    return predict_image(read_rgb(path))
