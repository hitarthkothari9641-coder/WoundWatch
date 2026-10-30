"""Score an image through quality gate, segmentation and calibrated classifier."""
import argparse,json
from .common import read_rgb
from .inference import predict_ssi,load_segmenter,quality_check

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("image"); ap.add_argument("--quality-only",action="store_true"); a=ap.parse_args()
    im=read_rgb(a.image)
    if a.quality_only: result=quality_check(im,load_segmenter()[0])
    else: result=predict_ssi(a.image)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
