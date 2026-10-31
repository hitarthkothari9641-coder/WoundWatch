import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from service.app import app
from service.security import configuration

class SecurityTests(unittest.TestCase):
    def test_production_fails_closed(self):
        with patch.dict(os.environ,{'WOUNDWATCH_ENV':'production','WOUNDWATCH_USERNAME':'','WOUNDWATCH_PASSWORD':''}):
            with self.assertRaises(RuntimeError): configuration()
        with patch.dict(os.environ,{'WOUNDWATCH_ENV':'prod'}):
            with self.assertRaises(RuntimeError): configuration()

    def test_auth_host_origin_and_size(self):
        with patch.dict(os.environ,{'WOUNDWATCH_USERNAME':'operator','WOUNDWATCH_PASSWORD':'test-only-password-123456789','WOUNDWATCH_ENV':'local'}):
            client=TestClient(app)
            self.assertEqual(client.get('/api/patients').status_code,401)
            self.assertEqual(client.get('/live').status_code,200)
            auth=('operator','test-only-password-123456789')
            self.assertEqual(client.get('/api/settings',auth=auth).status_code,200)
            self.assertEqual(client.get('/api/settings',auth=auth,headers={'host':'evil.example'}).status_code,400)
            self.assertEqual(client.post('/api/notifications/read',auth=auth,headers={'origin':'https://evil.example'}).status_code,403)
            self.assertEqual(client.post('/predict',auth=auth,headers={'content-length':'99999999'}).status_code,413)
            with patch.dict(os.environ,{'MAX_UPLOAD_BYTES':'1'}):
                self.assertEqual(client.post('/predict',auth=auth,content=iter([b'x'*40000,b'x'*40000])).status_code,413)
            with patch.dict(os.environ,{'WOUNDWATCH_ENV':'production','WOUNDWATCH_ALLOWED_HOSTS':'testserver'}):
                self.assertEqual(client.get('/api/settings',auth=auth).status_code,400)
                secure=TestClient(app,base_url='https://testserver')
                response=secure.get('/api/settings',auth=auth)
                self.assertEqual(response.status_code,200)
                self.assertIn('strict-transport-security',response.headers)

if __name__=='__main__': unittest.main()
