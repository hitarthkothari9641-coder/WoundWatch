"""Write reference Python outputs for service/test_parity.py."""
import argparse,csv
from pathlib import Path
from .inference import predict_ssi

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("images",nargs="+",help="Image paths"); ap.add_argument("--output",default="service/python_parity.csv"); a=ap.parse_args()
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["image_path","python_probability","python_status"]); writer.writeheader()
        for p in a.images:
            result=predict_ssi(p); writer.writerow({"image_path":str(Path(p).resolve()),"python_probability":"" if result["probability"] is None else result["probability"],"python_status":result["status"]})
    print(f"Saved reference records: {output}")
if __name__=="__main__": main()
