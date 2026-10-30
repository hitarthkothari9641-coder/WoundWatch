"""Export trained PyTorch classifier and segmenter to ONNX."""
import torch
from torch import nn
from .config import CFG
from .inference import load_classifier,load_segmenter

def main():
    CFG.create_dirs(); dev=torch.device("cpu")
    cl,ck=load_classifier(); cl=cl.to(dev).eval(); x=torch.zeros(1,3,*ck["image_size"])
    class Probabilities(nn.Module):
        def __init__(self,net): super().__init__(); self.net=net
        def forward(self,image): return torch.softmax(self.net(image),dim=1)
    torch.onnx.export(Probabilities(cl),x,CFG.models/"classifier.onnx",input_names=["image"],output_names=["probabilities"],opset_version=17,dynamic_axes={"image":{0:"batch"},"probabilities":{0:"batch"}},dynamo=False)
    sg,sk=load_segmenter(); sg=sg.to(dev).eval(); x=torch.zeros(1,3,*sk["image_size"])
    torch.onnx.export(sg,x,CFG.models/"segmenter.onnx",input_names=["image"],output_names=["logits"],opset_version=17,dynamic_axes={"image":{0:"batch"},"logits":{0:"batch"}},dynamo=False)
    print(f"Exported ONNX models to {CFG.models}")
if __name__=="__main__": main()
