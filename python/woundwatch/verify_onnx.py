"""Compare ONNX Runtime with PyTorch on up to 20 actual project images/model."""
import numpy as np,onnxruntime as ort,pandas as pd,torch
from .common import image_tensor,read_rgb,device
from .config import CFG
from .inference import load_classifier,load_segmenter

def verify(model,onnx_path,paths,size,probabilities=False):
    session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"]); errors=[]
    model.eval()
    for p in paths[:CFG.onnx_verification_images]:
        x=image_tensor(read_rgb(p),size).unsqueeze(0)
        with torch.inference_mode():
            out=model(x.to(device())); out=torch.softmax(out,1) if probabilities else out
            expected=out.cpu().numpy()
        got=session.run(None,{session.get_inputs()[0].name:x.numpy()})[0]
        errors.append(float(np.max(np.abs(expected-got))))
    if not errors: raise ValueError(f"No images available for ONNX parity at {onnx_path}")
    maximum=max(errors); print(f"{onnx_path.name}: tested {len(errors)} images; max absolute difference={maximum:.8g}")
    if maximum>CFG.onnx_parity_tolerance: raise RuntimeError(f"ONNX parity failed: maximum absolute difference {maximum:.8g} exceeds {CFG.onnx_parity_tolerance}")

def main():
    cl,ck=load_classifier(); sg,sk=load_segmenter()
    manifest=pd.read_csv(CFG.manifest); cls=[]
    for r in manifest.itertuples(index=False):
        p=CFG.surgwound/str(r.file)
        if not p.is_file(): p=CFG.surgwound/"images"/str(r.split)/p.name
        cls.append(p)
    seg=sorted((CFG.fuseg_external/"images").glob("*.png"))
    verify(cl,CFG.models/"classifier.onnx",cls,ck["image_size"],True)
    verify(sg,CFG.models/"segmenter.onnx",seg,sk["image_size"],False)
if __name__=="__main__": main()
