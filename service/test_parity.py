"""Compare Python checkpoint predictions with the running FastAPI service.

CSV columns: image_path,python_probability,python_status. Set PYTHON_PARITY_CSV
and WOUNDWATCH_URL before running `python -m unittest service.test_parity`.
"""
import csv,os,unittest
from pathlib import Path
import requests

class ServiceParityTest(unittest.TestCase):
    def test_probabilities_match_local_python(self):
        value=os.getenv("PYTHON_PARITY_CSV",""); self.assertTrue(value,"Set PYTHON_PARITY_CSV to saved Python reference outputs")
        csv_path=Path(value).resolve(); self.assertTrue(csv_path.is_file())
        url=os.getenv("WOUNDWATCH_URL","http://127.0.0.1:8000").rstrip("/"); tol=float(os.getenv("PARITY_TOLERANCE","1e-5"))
        session=requests.Session()
        login=session.post(url+'/api/auth/login',json={'email':os.getenv('WOUNDWATCH_TEST_EMAIL',''),'password':os.getenv('WOUNDWATCH_TEST_PASSWORD','')},timeout=30)
        self.assertEqual(login.status_code,200,'Set WOUNDWATCH_TEST_EMAIL and WOUNDWATCH_TEST_PASSWORD for a test account')
        session.headers['X-CSRF-Token']=login.json()['csrf_token']
        with csv_path.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
        self.assertTrue(rows,"Parity CSV has no records")
        for n,row in enumerate(rows,2):
            p=Path(row["image_path"]); p=p if p.is_absolute() else (csv_path.parent/p).resolve(); self.assertTrue(p.is_file(),f"row {n}: image missing")
            with p.open("rb") as image: res=session.post(url+"/predict",files={"file":(p.name,image,"application/octet-stream")},timeout=120)
            self.assertEqual(res.status_code,200,f"service failed on parity row {n}"); out=res.json()
            self.assertEqual(out["status"],row["python_status"])
            if row["python_probability"]:
                self.assertLessEqual(abs(float(out["probability"])-float(row["python_probability"])),tol)

if __name__=="__main__": unittest.main()
