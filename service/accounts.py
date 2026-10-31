"""Named accounts, revocable cookie sessions, and patient authorization."""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from typing import Literal
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from . import store

router=APIRouter(prefix='/api/auth')
identity=ContextVar('woundwatch_identity',default=None)
COOKIE='woundwatch_session'
SESSION_SECONDS=8*60*60

def password_hash(password,salt=None):
    salt=salt or secrets.token_hex(16)
    digest=hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),600000).hex()
    return f'pbkdf2_sha256$600000${salt}${digest}'

def check_password(password,encoded):
    try:
        _,_,salt,_=encoded.split('$')
        return hmac.compare_digest(password_hash(password,salt),encoded)
    except (ValueError,TypeError): return False

def actor():
    user=identity.get()
    if not user: raise HTTPException(401,'Please sign in to continue')
    return user

def clinician():
    user=actor()
    if user['role']!='clinician': raise HTTPException(403,'Only an assigned healthcare worker can perform this action')
    return user

def patient_ids():
    user=actor()
    if user['role']=='patient': return {user['patient_id']}
    with store.connection() as db:
        return {r[0] for r in db.execute('SELECT patient_id FROM assignments WHERE user_id=?',(user['id'],))}

def authorize(patient_id):
    if patient_id not in patient_ids(): raise HTTPException(404,'Record not found')

def audit(db,action,target,user_id=None):
    db.execute('INSERT INTO audit(actor,action,target,created_at) VALUES(?,?,?,?)',
        (user_id or (identity.get() or {}).get('id','administrator'),action,target,datetime.now(timezone.utc).isoformat()))

def public_user(row):
    return {k:row[k] for k in ('id','email','name','role','patient_id')}

def create_user(email,name,password,role='patient',patient_id=None):
    if role not in ('patient','clinician') or len(password)<12 or len(password)>128: raise ValueError('Use a valid role and a password of 12–128 characters')
    encoded=password_hash(password)
    with store.connection() as db:
        if patient_id and not db.execute('SELECT id FROM patients WHERE id=?',(patient_id,)).fetchone(): raise ValueError('Patient not found')
        if role=='patient' and not patient_id:
            patient_id='P-'+uuid4().hex[:10]
            p=dict(id=patient_id,name=name,age=None,gender='',contact='',surgery_type='',surgery_date='',
                   wound_location='',monitoring_status='monitoring',created_at=datetime.now(timezone.utc).isoformat())
            db.execute('INSERT INTO patients VALUES(?,?)',(patient_id,json.dumps(p)))
        uid=uuid4().hex
        db.execute('INSERT INTO users(id,email,name,password_hash,role,patient_id) VALUES(?,?,?,?,?,?)',
                   (uid,email.strip().lower(),name,encoded,role,patient_id))
        audit(db,'account_created',uid,uid)
        return dict(id=uid,email=email.strip().lower(),name=name,role=role,patient_id=patient_id)

def throttle(key):
    now=time.time(); key=hashlib.sha256(key.encode()).hexdigest()
    with store.connection() as db:
        db.execute('DELETE FROM auth_attempts WHERE reset_at<?',(now,))
        row=db.execute('SELECT count FROM auth_attempts WHERE key=?',(key,)).fetchone()
        if row and row[0]>=10: raise HTTPException(429,'Too many attempts. Please try again in 15 minutes.')
        db.execute('INSERT INTO auth_attempts VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1',(key,now+900))

def new_session(user,response):
    raw=secrets.token_urlsafe(32);csrf=secrets.token_urlsafe(32)
    with store.connection() as db:
        db.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
        db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(hashlib.sha256(raw.encode()).hexdigest(),user['id'],csrf,time.time()+SESSION_SECONDS))
        audit(db,'login',user['id'],user['id'])
    response.set_cookie(COOKIE,raw,max_age=SESSION_SECONDS,httponly=True,secure=os.getenv('WOUNDWATCH_ENV')=='production',samesite='strict',path='/')
    return {'user':user,'csrf_token':csrf}

class Credentials(BaseModel):
    model_config=ConfigDict(extra='forbid')
    email:str=Field(min_length=3,max_length=254)
    # Login must accept existing legacy/demo passwords; account creation and
    # password changes retain the 12-character minimum below.
    password:str=Field(min_length=1,max_length=128)
    @field_validator('email')
    @classmethod
    def email_valid(cls,value):
        value=value.strip().lower()
        if value.count('@')!=1 or '.' not in value.rsplit('@',1)[-1] or any(c.isspace() for c in value): raise ValueError('Enter a valid email address')
        return value

class Registration(Credentials):
    password:str=Field(min_length=12,max_length=128)
    name:str=Field(min_length=1,max_length=100)
    @field_validator('name')
    @classmethod
    def name_valid(cls,value):
        if not value.strip(): raise ValueError('Enter your name')
        return value.strip()

class LoginCredentials(Credentials):
    account_type: Literal['patient','clinician'] | None = None

@router.post('/register',status_code=201)
def register(body:Registration,request:Request,response:Response):
    throttle('register:'+(request.client.host if request.client else 'local'))
    try: user=create_user(body.email,body.name,body.password)
    except sqlite3.IntegrityError: raise HTTPException(409,'An account cannot be created with these details. Try signing in or contact the administrator.')
    return new_session(user,response)

@router.post('/login')
def login(body:LoginCredentials,request:Request,response:Response):
    throttle('login-ip:'+(request.client.host if request.client else 'local'))
    throttle('login-account:'+body.email)
    with store.connection() as db:
        row=db.execute('SELECT * FROM users WHERE email=?',(body.email,)).fetchone()
    encoded=row['password_hash'] if row else 'pbkdf2_sha256$600000$'+'00'*16+'$'+'00'*32
    valid=check_password(body.password,encoded)
    if not valid or not row or not row['active']: raise HTTPException(401,'Email or password is incorrect')
    if body.account_type and row['role']!=body.account_type: raise HTTPException(401,'Account type does not match. Choose Patient Login or Doctor Login.')
    return new_session(public_user(row),response)

@router.get('/me')
def me():
    user=actor()
    return {'user':{k:v for k,v in user.items() if k not in ('csrf','session_hash')},'csrf_token':user['csrf']}

@router.post('/logout')
def logout(response:Response):
    user=actor()
    with store.connection() as db:
        db.execute('DELETE FROM sessions WHERE token_hash=?',(user['session_hash'],))
    response.delete_cookie(COOKIE,path='/')
    return {'status':'ok'}

class PasswordChange(BaseModel):
    current_password:str=Field(max_length=128)
    new_password:str=Field(min_length=12,max_length=128)

@router.post('/password')
def change_password(body:PasswordChange,response:Response):
    user=actor();throttle('password:'+user['id'])
    with store.connection() as db:
        row=db.execute('SELECT password_hash FROM users WHERE id=?',(user['id'],)).fetchone()
        if not check_password(body.current_password,row[0]): raise HTTPException(400,'Current password is incorrect')
        db.execute('UPDATE users SET password_hash=? WHERE id=?',(password_hash(body.new_password),user['id']))
        db.execute('DELETE FROM sessions WHERE user_id=?',(user['id'],))
        audit(db,'password_changed',user['id'])
    response.delete_cookie(COOKIE,path='/')
    return {'status':'ok'}

def session_user(raw):
    if not raw or len(raw)>128:return None
    digest=hashlib.sha256(raw.encode()).hexdigest()
    with store.connection() as db:
        row=db.execute('SELECT u.*,s.csrf FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>? AND u.active=1',(digest,time.time())).fetchone()
    return {**public_user(row),'csrf':row['csrf'],'session_hash':digest} if row else None

class AccountMiddleware:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        path=scope['path']
        protected=path.startswith('/api/') or path in ('/predict','/health','/docs','/redoc','/openapi.json')
        if not protected or path in ('/api/auth/login','/api/auth/register'):
            return await self.app(scope,receive,send)
        headers={k.decode().lower():v.decode() for k,v in scope['headers']}
        cookie=SimpleCookie()
        try:cookie.load(headers.get('cookie',''))
        except Exception:pass
        raw=cookie[COOKIE].value if COOKIE in cookie else ''
        user=await run_in_threadpool(session_user,raw)
        if not user:
            return await JSONResponse({'detail':'Please sign in to continue'},status_code=401)(scope,receive,send)
        if scope['method'] not in ('GET','HEAD','OPTIONS') and not hmac.compare_digest(headers.get('x-csrf-token',''),user['csrf']):
            return await JSONResponse({'detail':'Session verification failed. Refresh the page and try again.'},status_code=403)(scope,receive,send)
        token=identity.set(user)
        try:await self.app(scope,receive,send)
        finally:identity.reset(token)
