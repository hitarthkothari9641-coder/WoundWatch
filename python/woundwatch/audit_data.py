"""Audit image counts, pixel dimensions, unreadable files and duplicates."""
import hashlib,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from .config import CFG

EXT={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
def ahash(image):
    a=np.asarray(image.convert("L").resize((8,8),Image.Resampling.LANCZOS),dtype=np.float32); v=a>=a.mean(); n=0
    for b in v.ravel(): n=(n<<1)|int(b)
    return n
def main():
    CFG.create_dirs(); rows=[]
    if CFG.manifest.is_file():
        df=pd.read_csv(CFG.manifest)
        if not {"file","split","label"}.issubset(df.columns): raise ValueError("SurgWound manifest requires file, split, label")
        for r in df.itertuples(index=False):
            rel=Path(str(r.file)); p=next((x for x in (rel,CFG.surgwound/rel,CFG.surgwound/"images"/str(r.split)/rel.name) if x.is_file()),CFG.surgwound/"images"/str(r.split)/rel.name)
            rows.append(["SurgWound",str(r.label),p])
    for ds,root in (("Kaggle",CFG.kaggle),("Medetec",CFG.medetec)):
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXT: rows.append([ds,p.relative_to(root).parts[0] if len(p.relative_to(root).parts)>1 else "Unlabelled",p])
    if not rows: raise FileNotFoundError("No images found under configured dataset paths")
    records=[]
    for ds,label,p in rows:
        r={"dataset":ds,"class":label,"file":str(p),"width":None,"height":None,"sha256":"","phash":"","unreadable":False,"error":""}
        try:
            data=p.read_bytes(); r["sha256"]=hashlib.sha256(data).hexdigest()
            with Image.open(p) as im: im.load(); r["width"],r["height"]=im.size; r["phash"]=f"{ahash(im):016x}"
        except Exception as e: r["unreadable"]=True; r["error"]=type(e).__name__
        records.append(r)
    tab=pd.DataFrame(records); tab["exact_duplicate_of"]=""; tab["near_duplicate_of"]=""
    for _,g in tab[~tab.unreadable].groupby("sha256"):
        if len(g)>1:
            for ix in g.index[1:]: tab.at[ix,"exact_duplicate_of"]=tab.at[g.index[0],"file"]
    # BK-tree lookup keeps perceptual-hash matching scalable and returns one
    # example partner per image rather than materializing all cluster pairs.
    tree={"value":None,"rows":[],"children":{}}
    def insert(node,value,row):
        if node["value"] is None: node.update(value=value,rows=[row]); return
        distance=(value^node["value"]).bit_count()
        if distance==0: node["rows"].append(row); return
        if distance not in node["children"]: node["children"][distance]={"value":value,"rows":[row],"children":{}}
        else: insert(node["children"][distance],value,row)
    def find_near(node,value,row):
        if node["value"] is None: return None
        distance=(value^node["value"]).bit_count()
        for other in node["rows"]:
            if other!=row and distance<=6 and not tab.at[other,"exact_duplicate_of"] and not tab.at[row,"exact_duplicate_of"]: return other
        for edge,child in node["children"].items():
            if distance-6<=edge<=distance+6:
                result=find_near(child,value,row)
                if result is not None: return result
        return None
    for ix in tab.index[~tab.unreadable]:
        val=int(tab.at[ix,"phash"],16); other=find_near(tree,val,ix)
        if other is not None:
            tab.at[ix,"near_duplicate_of"]=tab.at[other,"file"]
            if not tab.at[other,"near_duplicate_of"]: tab.at[other,"near_duplicate_of"]=tab.at[ix,"file"]
        insert(tree,val,ix)
    out=CFG.results/"data_audit.csv"; tab.to_csv(out,index=False); print(f"Audit rows={len(tab)}; saved {out}")
    for ds,g in tab.groupby("dataset",sort=False):
        print(f"{ds}: {len(g)} images; unreadable={int(g.unreadable.sum())}; exact duplicate rows={int(g.exact_duplicate_of.ne('').sum())}; near matches={int(g.near_duplicate_of.ne('').sum())}")
        for label,n in sorted(Counter(g["class"]).items()): print(f"  {label}: {n}")
        print("  common dimensions:",g.loc[~g.unreadable].groupby(["width","height"]).size().sort_values(ascending=False).head(5).to_dict())
    for p in tab.loc[tab.unreadable,"file"]: print(f"UNREADABLE: {p}")
if __name__=="__main__": main()
