"""Integration checks use actual checkpoints and isolated temporary records."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from fastapi.testclient import TestClient
from PIL import Image
from service.app import app, models_once
from service import store
from service.accounts import create_user
from woundwatch.inference import predict_image
from woundwatch.common import read_rgb

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original = store.ROOT
        store.ROOT = Path(cls.temp.name)
        cls.client = TestClient(app)
        create_user('dashboard-test@example.org','Test clinician','test-only-passphrase-123','clinician')
        login=cls.client.post('/api/auth/login',json={'email':'dashboard-test@example.org','password':'test-only-passphrase-123'})
        cls.client.headers['X-CSRF-Token']=login.json()['csrf_token']
        cls.sample = Path('data/raw/surgwound/images/train/02b44effbbf2.jpg')
        cls.patient_body = dict(name='Integration research alias', age=42, surgery_type='Research test',
                                surgery_date='2026-01-01', wound_location='Test location', monitoring_status='monitoring')

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        store.ROOT = cls.original
        cls.temp.cleanup()

    def patient(self):
        r = self.client.post('/api/patients', json=self.patient_body)
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def upload(self, patient, previous='', content=None):
        return self.client.post('/api/scans', data=dict(patient_id=patient['id'],scan_date='2026-01-02',
            surgery_date='2026-01-01',wound_location='Test location',previous_id=previous),
            files={'file': ('scan.jpg', content if content is not None else self.sample.read_bytes(), 'image/jpeg')})

    def test_actual_model_parity_and_persistence(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        (seg,_),(clf,ck) = models_once()
        expected = predict_image(read_rgb(self.sample), seg, clf, ck)
        self.assertIsNotNone(expected['probability'], 'Fixture must exercise classifier, not only quality gate')
        r = self.client.post('/predict', files={'file': ('scan.jpg', self.sample.read_bytes())})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), expected)
        patient = self.patient()
        first = self.upload(patient)
        self.assertEqual(first.status_code, 201, first.text)
        scan = first.json()
        self.assertIsNotNone(scan['features'])
        self.assertEqual(scan['result'], expected)
        second = self.upload(patient, scan['id'])
        self.assertEqual(second.status_code, 201, second.text)
        other = self.patient()
        self.assertEqual(self.upload(other, scan['id']).status_code, 422)
        if scan['features']:
            self.assertTrue(all(abs(v)<1e-5 for v in second.json()['drift'].values()))
        self.assertEqual(len(self.client.get('/api/scans', params={'patient_id':patient['id']}).json()),2)
        image = self.client.get('/api/scans/'+scan['id']+'/image')
        self.assertEqual(image.status_code,200)
        self.assertEqual(Image.open(io.BytesIO(image.content)).format,'PNG')
        review = self.client.patch('/api/scans/'+scan['id']+'/review', json={'status':'reviewed'})
        self.assertEqual(review.json()['review_status'],'reviewed')
        self.assertTrue(self.client.get('/api/notifications').json())
        self.client.post('/api/notifications/read')
        self.assertTrue(all(n['read'] for n in self.client.get('/api/notifications').json()))
        with TestClient(app,cookies=self.client.cookies) as reopened:
            self.assertEqual(len(reopened.get('/api/scans',params={'patient_id':patient['id']}).json()),2)
        print('Actual checkpoint result:', expected, 'Colour features available:', bool(scan['features']))

    def test_patient_validation_and_edit(self):
        self.assertEqual(self.client.post('/api/patients',json={**self.patient_body,'name':' '}).status_code,422)
        self.assertEqual(self.client.post('/api/patients',json={**self.patient_body,'age':-1}).status_code,422)
        p=self.patient()
        r=self.client.put('/api/patients/'+p['id'],json={**self.patient_body,'name':'Edited alias'})
        self.assertEqual(r.json()['name'],'Edited alias')

    def test_upload_rejections_and_model_failure(self):
        p=self.patient()
        self.assertEqual(self.upload(p,content=b'not an image').status_code,400)
        with patch('service.app.MAX_BYTES',10):
            self.assertEqual(self.upload(p,content=b'x'*11).status_code,413)
        out=io.BytesIO();Image.new('RGB',(10,10)).save(out,format='GIF')
        self.assertEqual(self.upload(p,content=out.getvalue()).status_code,415)
        with patch('service.app.models_once',side_effect=RuntimeError('private detail')):
            r=self.upload(p)
            self.assertEqual(r.status_code,503)
            self.assertNotIn('private detail',r.text)
        self.assertEqual(self.upload(p,previous='missing').status_code,404)
        self.assertEqual(self.client.get('/api/scans/missing/image').status_code,404)

    def test_quality_retake(self):
        out=io.BytesIO();Image.new('RGB',(64,64),(120,100,90)).save(out,format='PNG')
        response=self.upload(self.patient(),content=out.getvalue())
        self.assertEqual(response.status_code,201,response.text)
        s=response.json()
        self.assertEqual(s['result']['status'],'retake')
        self.assertIsNone(s['result']['probability'])
        self.assertEqual(s['stages']['classification'],'skipped')
        response=self.client.get(f"/api/scans/{s['id']}/visualization/gradcam")
        self.assertEqual(response.status_code,422)
        self.assertEqual(self.client.get(f"/api/scans/{s['id']}/visualization/colour").status_code,200)

    def test_visualizations_and_relink(self):
        p=self.patient();a=self.upload(p).json();b=self.upload(p).json()
        self.assertIsNone(b['drift'])
        response=self.client.patch(f"/api/scans/{b['id']}/comparison",json={'previous_id':a['id']})
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(all(abs(v)<1e-5 for v in response.json()['drift'].values()))
        self.assertEqual(self.client.patch(f"/api/scans/{a['id']}/comparison",json={'previous_id':b['id']}).status_code,422)
        for kind in ('colour','gradcam'):
            response=self.client.get(f"/api/scans/{b['id']}/visualization/{kind}")
            self.assertEqual(response.status_code,200,response.text[:200] if response.status_code!=200 else '')
            self.assertEqual(Image.open(io.BytesIO(response.content)).format,'PNG')
        response=self.client.post(f"/api/scans/{a['id']}/colour")
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['result'],a['result'])
        # Model state must be unchanged by gradient computation.
        again=self.upload(p).json()
        self.assertEqual(again['result'],a['result'])

    def test_frontend_assets(self):
        for path in ['/','/assets/app.js','/assets/style.css']:
            r=self.client.get(path)
            self.assertEqual(r.status_code,200)
            self.assertEqual(r.headers['cache-control'],'no-store')

if __name__=='__main__': unittest.main()
