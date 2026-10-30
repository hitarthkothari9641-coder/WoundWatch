"""Small CPU unit tests for input validation and Python inference output contracts."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from PIL import Image
from woundwatch.common import image_tensor,read_rgb
from woundwatch.inference import predict_image,quality_check

class HalfMask(nn.Module):
    def forward(self,x):
        b,_,h,w=x.shape; out=torch.zeros((b,2,h,w),device=x.device); out[:,1,:,:w//2]=4; return out
class FixedClassifier(nn.Module):
    def forward(self,x): return torch.tensor([[0.,1.]],device=x.device).repeat(x.shape[0],1)

class PipelineTests(unittest.TestCase):
    def test_bad_path(self):
        with self.assertRaises(ValueError): read_rgb("missing_woundwatch_test_image.jpg")
    def test_grayscale_becomes_rgb_tensor(self):
        im=Image.new("L",(17,11),128); tensor=image_tensor(im,(32,32)); self.assertEqual(tuple(tensor.shape),(3,32,32))
    def test_tiny_image_quality_returns_reasons_without_crashing(self):
        im=Image.new("RGB",(2,2),(127,127,127)); result=quality_check(im,HalfMask())
        self.assertIn("passed",result); self.assertGreaterEqual(result["wound_area_fraction"],0)
    def test_prediction_contract_and_probability_range(self):
        im=Image.new("RGB",(32,24),(120,100,90)); ck={"image_size":[32,32],"platt_intercept":0.,"platt_slope":1.,"threshold_youden":.5,"defer_half_width":0.01}
        result=predict_image(im,HalfMask(),FixedClassifier(),ck)
        self.assertIn(result["status"],["ok","defer","retake"])
        if result["probability"] is not None: self.assertGreaterEqual(result["probability"],0); self.assertLessEqual(result["probability"],1)

if __name__=="__main__": unittest.main()
