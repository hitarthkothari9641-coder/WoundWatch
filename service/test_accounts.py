import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from service.app import app
from service import store
from service.accounts import create_user

PASSWORD='isolated-test-password-123'

def signed_client(email):
    client=TestClient(app)
    result=client.post('/api/auth/login',json={'email':email,'password':PASSWORD})
    if result.status_code!=200:raise AssertionError(result.text)
    client.headers['X-CSRF-Token']=result.json()['csrf_token']
    return client

class AccountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.original=store.ROOT;store.ROOT=Path(cls.temp.name)
        cls.a=create_user('a@example.org','Patient A',PASSWORD)
        cls.b=create_user('b@example.org','Patient B',PASSWORD)
        cls.doc=create_user('doctor@example.org','Assigned clinician',PASSWORD,'clinician')
        cls.other=create_user('other@example.org','Other clinician',PASSWORD,'clinician')
        with store.connection() as db:db.execute('INSERT INTO assignments VALUES(?,?)',(cls.doc['id'],cls.a['patient_id']))
        cls.ca=signed_client('a@example.org');cls.cb=signed_client('b@example.org');cls.cd=signed_client('doctor@example.org');cls.co=signed_client('other@example.org')
        cls.wound=cls.ca.post(f"/api/patients/{cls.a['patient_id']}/wounds",json={'surgery_date':'2026-01-01','surgery_type':'Test surgery','wound_location':'Left knee'}).json()
        cls.sample=Path('data/raw/surgwound/images/train/02b44effbbf2.jpg').read_bytes()
        response=cls.ca.post('/api/scans',data={'patient_id':cls.a['patient_id'],'scan_date':'2026-01-02','surgery_date':'2026-01-01','wound_location':'Left knee','wound_id':cls.wound['id']},files={'file':('image.jpg',cls.sample,'image/jpeg')})
        if response.status_code!=201:raise AssertionError(response.text)
        cls.scan=response.json()

    @classmethod
    def tearDownClass(cls):
        for c in (cls.ca,cls.cb,cls.cd,cls.co):c.close()
        store.ROOT=cls.original;cls.temp.cleanup()

    def test_role_and_object_isolation(self):
        a=self.a['patient_id'];sid=self.scan['id']
        self.assertEqual([p['id'] for p in self.ca.get('/api/patients').json()],[a])
        self.assertEqual([p['id'] for p in self.cd.get('/api/patients').json()],[a])
        self.assertEqual(self.co.get('/api/patients').json(),[])
        self.assertEqual(self.cb.get('/api/scans').json(),[])
        for client in (self.cb,self.co):
            for route in (f'/api/scans?patient_id={a}',f'/api/patients/{a}/wounds',f'/api/scans/{sid}/image',f'/api/scans/{sid}/report.pdf',f'/api/scans/{sid}/visualization/colour',f'/api/scans/{sid}/visualization/gradcam'):
                self.assertEqual(client.get(route).status_code,404,route)
            self.assertEqual(client.post(f'/api/scans/{sid}/reanalyse').status_code,404)
            self.assertEqual(client.post(f'/api/scans/{sid}/colour').status_code,404)
            self.assertEqual(client.patch(f'/api/scans/{sid}/comparison',json={'previous_id':None}).status_code,404)
            self.assertEqual(client.get('/api/notifications').json(),[])
        self.assertEqual(self.ca.patch(f'/api/scans/{sid}/review',json={'status':'reviewed'}).status_code,403)
        self.assertEqual(self.cd.patch(f'/api/scans/{sid}/review',json={'status':'reviewed'}).status_code,200)
        self.assertEqual(self.ca.patch(f'/api/patients/{a}/monitoring',json={'monitoring_status':'discharged'}).status_code,403)
        self.assertEqual(self.co.patch(f'/api/patients/{a}/monitoring',json={'monitoring_status':'discharged'}).status_code,404)
        self.assertEqual(self.cd.patch(f'/api/patients/{a}/monitoring',json={'monitoring_status':'follow-up'}).status_code,200)
        self.assertEqual(self.ca.post('/api/patients',json={'name':'No','surgery_type':'No','surgery_date':'2026-01-01','wound_location':'No'}).status_code,403)

    def test_profile_wounds_and_cross_wound_comparison(self):
        pid=self.a['patient_id']
        response=self.ca.patch(f'/api/patients/{pid}/profile',json={'name':'Updated alias','age':45,'gender':'Not specified','contact':'Research contact'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.ca.patch(f'/api/patients/{pid}/profile',json={'name':'No','role':'clinician'}).status_code,422)
        other=self.ca.post(f'/api/patients/{pid}/wounds',json={'surgery_date':'2026-01-01','surgery_type':'Test','wound_location':'Right knee'}).json()
        response=self.ca.post('/api/scans',data={'patient_id':pid,'scan_date':'2026-01-03','surgery_date':'2026-01-01','wound_location':'Right knee','wound_id':other['id'],'previous_id':self.scan['id']},files={'file':('x.jpg',self.sample)})
        self.assertEqual(response.status_code,422)
        response=self.cb.post('/api/scans',data={'patient_id':pid,'scan_date':'2026-01-02','surgery_date':'2026-01-01','wound_location':'Left knee'},files={'file':('x.jpg',self.sample)})
        self.assertEqual(response.status_code,404)

    def test_legacy_reanalysis_and_pdf(self):
        sid=self.scan['id']
        with store.connection() as db:
            original=json.loads(db.execute('SELECT payload FROM scans WHERE id=?',(sid,)).fetchone()[0])
            original.pop('model_fingerprints',None)
            db.execute('UPDATE scans SET payload=? WHERE id=?',(json.dumps(original),sid))
        self.assertEqual(self.ca.get(f'/api/scans/{sid}/visualization/gradcam').status_code,409)
        fresh=self.ca.post(f'/api/scans/{sid}/reanalyse')
        self.assertEqual(fresh.status_code,201,fresh.text)
        self.assertNotEqual(fresh.json()['id'],sid)
        self.assertEqual(store.get('scans',sid),original)
        self.assertEqual(self.ca.get(f"/api/scans/{fresh.json()['id']}/visualization/gradcam").status_code,200)
        response=self.ca.get(f'/api/scans/{sid}/report.pdf')
        self.assertEqual(response.status_code,200,response.text[:200] if response.status_code!=200 else '')
        self.assertTrue(response.content.startswith(b'%PDF'))

    def test_registration_csrf_expiry_and_logout(self):
        client=TestClient(app)
        self.assertEqual(client.get('/api/patients').status_code,401)
        response=client.post('/api/auth/register',json={'email':'new@example.org','name':'New patient','password':PASSWORD,'role':'clinician'})
        self.assertEqual(response.status_code,422)
        response=client.post('/api/auth/register',json={'email':'new@example.org','name':'New patient','password':PASSWORD})
        self.assertEqual(response.status_code,201,response.text)
        self.assertIn('HttpOnly',response.headers['set-cookie']);self.assertIn('SameSite=strict',response.headers['set-cookie'])
        self.assertEqual(response.json()['user']['role'],'patient')
        self.assertEqual(client.post('/api/auth/logout').status_code,403)
        client.headers['X-CSRF-Token']=response.json()['csrf_token']
        self.assertEqual(client.post('/api/auth/logout').status_code,200)
        self.assertEqual(client.get('/api/scans').status_code,401)
        login=client.post('/api/auth/login',json={'email':'new@example.org','password':PASSWORD})
        self.assertEqual(login.status_code,200)
        with store.connection() as db:db.execute('UPDATE sessions SET expires=? WHERE user_id=?',(time.time()-1,response.json()['user']['id']))
        self.assertEqual(client.get('/api/auth/me').status_code,401)

    def test_revocation_takes_effect_on_existing_session(self):
        with store.connection() as db:db.execute('INSERT OR IGNORE INTO assignments VALUES(?,?)',(self.other['id'],self.a['patient_id']))
        self.assertEqual(self.co.get(f"/api/scans/{self.scan['id']}/image").status_code,200)
        with store.connection() as db:db.execute('DELETE FROM assignments WHERE user_id=?',(self.other['id'],))
        self.assertEqual(self.co.get(f"/api/scans/{self.scan['id']}/image").status_code,404)

    def test_password_change_revokes_all_sessions(self):
        create_user('password@example.org','Password test',PASSWORD)
        first=signed_client('password@example.org');second=signed_client('password@example.org')
        updated='replacement-test-password-456'
        try:
            self.assertEqual(first.post('/api/auth/password',json={'current_password':'wrong','new_password':updated}).status_code,400)
            self.assertEqual(second.get('/api/auth/me').status_code,200)
            self.assertEqual(first.post('/api/auth/password',json={'current_password':PASSWORD,'new_password':updated}).status_code,200)
            for client in (first,second):
                self.assertEqual(client.get('/api/auth/me').status_code,401)
            self.assertEqual(first.post('/api/auth/login',json={'email':'password@example.org','password':PASSWORD}).status_code,401)
            self.assertEqual(first.post('/api/auth/login',json={'email':'password@example.org','password':updated}).status_code,200)
        finally:
            first.close();second.close()

if __name__=='__main__':unittest.main()
