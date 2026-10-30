"""Extract wound/ring CIELAB summaries and per-patient change from day 1."""
import argparse
import numpy as np
import pandas as pd
from .common import read_rgb
from .config import CFG
from .inference import load_segmenter,segment_wound,wound_and_ring

def rgb_to_cielab(image):
    """Convert sRGB to CIE L*a*b* using D65/2-degree reference white."""
    rgb=np.asarray(image.convert("RGB"),dtype=np.float64)/255.0
    linear=np.where(rgb<=0.04045,rgb/12.92,((rgb+0.055)/1.055)**2.4)
    xyz=linear @ np.array([[0.4124564,0.3575761,0.1804375],
                           [0.2126729,0.7151522,0.0721750],
                           [0.0193339,0.1191920,0.9503041]]).T
    xyz=xyz/np.array([0.95047,1.0,1.08883])
    delta=6/29; f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    fx,fy,fz=f[...,0],f[...,1],f[...,2]
    return np.stack((116*fy-16,500*(fx-fy),200*(fy-fz)),axis=-1).astype(np.float32)

def feature_row(image,mask,ring_pixels=12):
    wound,ring=wound_and_ring(image,mask,ring_pixels)
    lab=rgb_to_cielab(image)
    if not wound.any() or not ring.any(): raise ValueError("Wound or surrounding skin ring is empty")
    row={}
    for name,index in (("L",0),("a",1),("b",2)):
        w=lab[...,index][wound]; r=lab[...,index][ring]
        row[f"{name}_wound_mean"]=float(w.mean()); row[f"{name}_wound_median"]=float(np.median(w))
        row[f"{name}_ring_mean"]=float(r.mean()); row[f"{name}_ring_median"]=float(np.median(r))
    row["erythema_a_mean_difference"]=row["a_wound_mean"]-row["a_ring_mean"]
    row["erythema_a_median_difference"]=row["a_wound_median"]-row["a_ring_median"]
    return row

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("table",help="CSV with patientID, day, file"); ap.add_argument("--output",default=None); a=ap.parse_args()
    tab=pd.read_csv(a.table)
    if not {"patientID","day","file"}.issubset(tab.columns): raise ValueError("Input table needs patientID, day, file")
    seg,_=load_segmenter(); rows=[]
    for r in tab.itertuples(index=False):
        p=CFG.root/str(r.file) if not str(r.file).startswith(str(CFG.root)) else str(r.file)
        im=read_rgb(p); mask=segment_wound(im,seg)
        rows.append({"patientID":r.patientID,"day":r.day,"file":str(r.file),**feature_row(im,mask,CFG.ring_dilation_pixels)})
    result=pd.DataFrame(rows).sort_values(["patientID","day"])
    numeric=[c for c in result.select_dtypes(include="number").columns if c!="day"]
    baseline=result[result.day==1].set_index("patientID")[numeric]
    missing=set(result.patientID)-set(baseline.index)
    if missing: raise ValueError(f"Day 1 baseline is missing for {len(missing)} patient(s)")
    for c in numeric: result[f"{c}_drift_from_day1"]=result[c]-result.patientID.map(baseline[c])
    dest=a.output or str(CFG.results/"color_features.csv"); result.to_csv(dest,index=False); print(f"Saved color features: {dest}")

if __name__=="__main__": main()
