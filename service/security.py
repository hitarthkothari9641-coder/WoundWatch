"""Deployment guards for a single-operator research service."""
import base64
import hmac
import os
from urllib.parse import urlsplit
from starlette.responses import JSONResponse

def configuration():
    mode=os.getenv('WOUNDWATCH_ENV','local')
    if mode not in ('local','production'): raise RuntimeError('WOUNDWATCH_ENV must be local or production')
    production=mode=='production'
    username=os.getenv('WOUNDWATCH_USERNAME','')
    password=os.getenv('WOUNDWATCH_PASSWORD','')
    configured_hosts=[h.strip() for h in os.getenv('WOUNDWATCH_ALLOWED_HOSTS','').split(',') if h.strip()]
    render_host=os.getenv('RENDER_EXTERNAL_HOSTNAME','').strip()
    hosts=configured_hosts + ([render_host] if render_host else [])
    if not production:
        hosts += ['localhost','127.0.0.1','::1','testserver']
    if production and (not hosts or '*' in hosts):
        raise RuntimeError('Production requires explicit hosts or Render external hostname; wildcards are not allowed')
    if bool(username)!=bool(password): raise RuntimeError('Set both WOUNDWATCH_USERNAME and WOUNDWATCH_PASSWORD')
    return production,username,password,hosts

class SecurityMiddleware:
    def __init__(self,app): self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type']!='http': return await self.app(scope,receive,send)
        production,username,password,hosts=configuration()
        headers={k.decode('latin1').lower():v.decode('latin1') for k,v in scope['headers']}
        host=headers.get('host','')
        async def reject(status,message,extra=None):
            return await JSONResponse({'detail':message},status_code=status,headers={'Cache-Control':'no-store',**(extra or {})})(scope,receive,send)
        try: hostname=urlsplit('//'+host).hostname
        except ValueError: return await reject(400,'Invalid host')
        if hostname not in hosts: return await reject(400,'Host not allowed')
        # Render terminates TLS at its edge. Its internal health probe can use
        # HTTP, so keep only the non-sensitive liveness path available there.
        if production and scope.get('scheme')!='https' and scope['path']!='/live':
            return await reject(400,'HTTPS is required')
        if scope['path']!='/live' and username:
            valid=False
            try:
                scheme,encoded=headers.get('authorization','').split(' ',1)
                supplied_user,supplied_password=base64.b64decode(encoded,validate=True).decode('utf-8').split(':',1)
                valid=(scheme.lower()=='basic' and hmac.compare_digest(supplied_user.encode(),username.encode())
                       and hmac.compare_digest(supplied_password.encode(),password.encode()))
            except (ValueError,UnicodeError): pass
            if not valid: return await reject(401,'Authentication required',{'WWW-Authenticate':'Basic realm="WoundWatch", charset="UTF-8"'})
        if scope['method'] not in ('GET','HEAD','OPTIONS'):
            origin=headers.get('origin')
            if headers.get('sec-fetch-site')=='cross-site' or (origin and origin!=f"{scope.get('scheme')}://{host}"):
                return await reject(403,'Cross-origin writes are not allowed')
        limit=int(os.getenv('MAX_UPLOAD_BYTES',10*1024*1024))+65536
        try:
            if int(headers.get('content-length','0'))>limit: return await reject(413,'Request exceeds size limit')
        except ValueError: return await reject(400,'Invalid content length')
        # Enforce the body limit for chunked requests too, before multipart parsing.
        content=bytearray();size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect': return
            body=message.get('body',b'');size+=len(body)
            if size>limit: return await reject(413,'Request exceeds size limit')
            content.extend(body)
            if not message.get('more_body',False): break
        delivered=False
        async def bounded_receive():
            nonlocal delivered
            if delivered: return await receive()
            delivered=True
            return {'type':'http.request','body':bytes(content),'more_body':False}
        async def secured_send(message):
            if message['type']=='http.response.start':
                message['headers']=[(k,v) for k,v in message['headers'] if k.lower()!=b'cache-control']
                message['headers'].append((b'cache-control',b'no-store'))
                if production: message['headers'].append((b'strict-transport-security',b'max-age=31536000'))
            await send(message)
        await self.app(scope,bounded_receive,secured_send)
