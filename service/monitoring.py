"""Patient profiles, wound identities and private scan reports."""
import io
import json
import threading
import textwrap
from datetime import date
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL
from fastapi import APIRouter,HTTPException
from fastapi.responses import Response
from pydantic import BaseModel,ConfigDict,Field,field_validator
from . import store
from .accounts import authorize,actor,audit,clinician
from typing import Literal
from .dashboard import required,now

router=APIRouter(prefix='/api')
pdf_lock=threading.Lock()

def wound_key(patient_id,surgery_date,location):
    return 'W-'+uuid5(NAMESPACE_URL,json.dumps([patient_id,surgery_date,location.strip().casefold()])).hex[:20]

def enrich_scan(scan):
    return {**scan,'wound_id':wound_key(scan['patient_id'],scan['surgery_date'],scan['wound_location'])}

def patient_wounds(patient_id):
    patient=required('patients',patient_id)
    wounds={}
    for s in store.records('scans'):
        if s['patient_id']==patient_id:
            key=wound_key(patient_id,s['surgery_date'],s['wound_location'])
            wounds[key]=dict(id=key,patient_id=patient_id,surgery_date=s['surgery_date'],wound_location=s['wound_location'],surgery_type=s.get('patient_snapshot',{}).get('surgery_type',''),created_at=s['created_at'])
    if patient.get('surgery_date') and patient.get('wound_location'):
        key=wound_key(patient_id,patient['surgery_date'],patient['wound_location'])
        wounds.setdefault(key,dict(id=key,patient_id=patient_id,surgery_date=patient['surgery_date'],wound_location=patient['wound_location'],surgery_type=patient.get('surgery_type',''),created_at=patient.get('created_at')))
    for w in store.records('wounds'):
        if w['patient_id']==patient_id:wounds[w['id']]=w
    return list(wounds.values())

def get_wound(patient_id,wound_id):
    return next((w for w in patient_wounds(patient_id) if w['id']==wound_id),None) or missing()

def missing():raise HTTPException(404,'Wound not found')

class Profile(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    name:str=Field(min_length=1,max_length=100)
    age:int|None=Field(default=None,ge=0,le=120)
    gender:str=Field(default='',max_length=40)
    contact:str=Field(default='',max_length=150)

@router.patch('/patients/{identifier}/profile')
def edit_profile(identifier:str,body:Profile):
    p=required('patients',identifier);p.update(body.model_dump());store.save('patients',p)
    with store.connection() as db:
        db.execute('UPDATE users SET name=? WHERE patient_id=?',(body.name,identifier))
        audit(db,'profile_updated',identifier)
    return p

class CareTeam(BaseModel):
    clinician_ids: list[str] = Field(max_length=50)

def _own_patient():
    user=actor()
    if user['role']!='patient': raise HTTPException(403,'Only patients choose who can see their record')
    return user['patient_id']

@router.get('/care-team')
def care_team():
    pid=_own_patient()
    with store.connection() as db:
        shared={r[0] for r in db.execute('SELECT user_id FROM assignments WHERE patient_id=?',(pid,))}
        return [dict(id=r[0],name=r[1],shared=r[0] in shared) for r in db.execute("SELECT id,name FROM users WHERE role='clinician' ORDER BY name")]

@router.put('/care-team')
def set_care_team(body:CareTeam):
    """Consent: the patient decides which doctors can see their record. Doctors see changes on their next refresh."""
    pid=_own_patient()
    with store.connection() as db:
        valid={r[0] for r in db.execute("SELECT id FROM users WHERE role='clinician'")}
        chosen=set(body.clinician_ids)&valid
        db.execute('DELETE FROM assignments WHERE patient_id=?',(pid,))
        db.executemany('INSERT INTO assignments VALUES(?,?)',[(c,pid) for c in chosen])
        audit(db,'care_team_updated',pid)
    return care_team()

class Wound(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    wound_location:str=Field(min_length=1,max_length=100)
    surgery_type:str=Field(min_length=1,max_length=100)
    surgery_date:date
    @field_validator('surgery_date')
    @classmethod
    def valid_date(cls,value):
        if value>date.today():raise ValueError('Surgery date cannot be in the future')
        return value

class MonitoringStatus(BaseModel):
    model_config=ConfigDict(extra='forbid')
    monitoring_status:Literal['monitoring','follow-up','discharged']

@router.patch('/patients/{identifier}/monitoring')
def update_monitoring(identifier:str,body:MonitoringStatus):
    clinician();p=required('patients',identifier);p['monitoring_status']=body.monitoring_status;store.save('patients',p)
    with store.connection() as db:audit(db,'monitoring_status_updated',identifier)
    return p

@router.get('/patients/{identifier}/wounds')
def wounds(identifier:str):return patient_wounds(identifier)

@router.post('/patients/{identifier}/wounds',status_code=201)
def add_wound(identifier:str,body:Wound):
    required('patients',identifier)
    key=wound_key(identifier,body.surgery_date.isoformat(),body.wound_location)
    if any(w['id']==key for w in patient_wounds(identifier)):raise HTTPException(409,'This wound and surgery are already recorded')
    record=dict(id=key,patient_id=identifier,created_at=now(),**body.model_dump(mode='json'))
    with store.connection() as db:
        db.execute('INSERT INTO wounds VALUES(?,?,?)',(key,identifier,json.dumps(record)))
        audit(db,'wound_created',key)
    return record

ASSETS=Path(__file__).parent/'report_assets'
NAVY,TEAL,SINDOOR,KESARI,HALDI='#0C1E36','#0E8388','#D92638','#E25822','#F59E0B'
BAND_COLOUR={'elevated':SINDOOR,'defer':HALDI,'low':TEAL}
A4=(8.27,11.69)

def _font(name,size):
    from matplotlib.font_manager import FontProperties
    return FontProperties(fname=str(ASSETS/name),size=size)

def _text(fig,x,y,s,font,size,color=NAVY,**kw):
    fig.text(x,y,s,fontproperties=_font(font,size),color=color,parse_math=False,**kw)

def _para(fig,x,y,s,font,size,width,step,color=NAVY):
    for seg in textwrap.wrap(s,width) or ['']:
        _text(fig,x,y,seg,font,size,color);y-=step
    return y

def _page(title,subtitle):
    """Brand page chrome: clinical seal (kit: 72-96px on audit reports), Outfit title, Sindoor/Peacock rule."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from PIL import Image
    fig=Figure(figsize=A4)
    with Image.open(ASSETS/'clinical-seal.png') as seal:
        ax=fig.add_axes([.07,.895,.115,.115*A4[0]/A4[1]]);ax.imshow(seal);ax.axis('off')
    _text(fig,.205,.947,title,'Outfit-ExtraBold.ttf',21);_text(fig,.205,.921,subtitle,'PlusJakartaSans-SemiBold.ttf',9.5,TEAL)
    fig.add_artist(Line2D([.07,.93],[.885,.885],color=TEAL,lw=1.2,transform=fig.transFigure))
    fig.add_artist(Line2D([.07,.16],[.885,.885],color=SINDOOR,lw=3,transform=fig.transFigure))
    return fig

def _section(fig,y,label):
    _text(fig,.07,y,label.upper(),'PlusJakartaSans-SemiBold.ttf',8,TEAL);return y-.028

def report_figures(scan,history,guide,original,overlays):
    snap=scan['patient_snapshot'];result=scan['result'];demo=bool(scan.get('demo_only'))
    prob=f"{100*result['probability']:.1f}%" if result['probability'] is not None else 'Unavailable'
    rows=[('Patient',f"{snap['name']} ({scan['patient_id']})"),('Surgery',f"{snap.get('surgery_type','Not recorded')} | {scan['surgery_date']}"),
          ('Wound',scan['wound_location']),('Photograph date',scan['scan_date']),('Analysis time',scan['created_at']),
          ('Screening band',f"{result['risk_band']} | workflow: {result['status']}"),('Elevated-label probability',prob),
          ('Healthcare review',f"{scan['review_status']} (recording review is not diagnosis approval)")]
    figs=[]
    fig=_page('Monitoring report','Post-surgical wound monitoring | Wound Trace');y=.85
    y=_section(fig,y,'Scan details')
    for k,v in rows:
        _text(fig,.07,y,k,'PlusJakartaSans-SemiBold.ttf',9)
        y=_para(fig,.31,y,v,'PlusJakartaSans-Regular.ttf',9.5,64,.022)-.006
    band=BAND_COLOUR.get(result['risk_band'],NAVY);y-=.012
    from matplotlib.patches import Rectangle
    box=textwrap.wrap(guide['summary'],70);h=.03+.022*len(box)
    fig.add_artist(Rectangle((.07,y-h+.012),.86,h,transform=fig.transFigure,facecolor=band,alpha=.09,edgecolor='none'))
    fig.add_artist(Rectangle((.07,y-h+.012),.007,h,transform=fig.transFigure,facecolor=band,edgecolor='none'))
    yy=y-.012
    for seg in box:_text(fig,.095,yy,seg,'PlusJakartaSans-SemiBold.ttf',10.5);yy-=.022
    y-=h+.02
    if demo:
        y=_para(fig,.07,y,'DEMO DATA ONLY: synthetic account and Kaggle wound-type example. Infection-risk prediction is withheld; not for clinical care.','PlusJakartaSans-SemiBold.ttf',8.5,95,.02,KESARI)-.01
    ax=fig.add_axes([.1,.1,.8,max(.1,y-.12)]);ax.imshow(original);ax.axis('off')
    figs.append(fig)

    fig=_page('How to read this scan','Guidance for the reviewing clinician');y=.85
    y=_section(fig,y,'What the model and measurements show')
    for line in guide['observations']:
        y=_para(fig,.09,y,'- '+line,'PlusJakartaSans-Regular.ttf',9.5,92,.021)-.008
    y=_section(fig,y-.01,'What a clinician could check')
    for line in guide['check']:
        y=_para(fig,.09,y,'- '+line,'PlusJakartaSans-Regular.ttf',9.5,92,.021)-.008
    y=_section(fig,y-.01,'Limits')
    notes=['Research screening only. This report does not confirm or exclude infection and does not replace clinical examination.',
           'Differences in colour can reflect lighting, white balance, skin tone, pose or segmentation error. A single image cannot establish healing change.']
    notes+=([f"Source category: {scan.get('source_class','unknown')}. Wound-type category is not an infection outcome."] if demo else
            ['The probability is a model estimate for the training label, not diagnostic certainty.'])
    for line in notes:y=_para(fig,.09,y,'- '+line,'PlusJakartaSans-Regular.ttf',9.5,92,.021)-.008
    figs.append(fig)

    for title,overlay in overlays:
        fig=_page('Visual analysis','Overlay for clinician review')
        ax=fig.add_axes([.1,.2,.8,.64]);ax.imshow(overlay);ax.axis('off')
        _para(fig,.07,.165,title,'PlusJakartaSans-SemiBold.ttf',9.5,95,.021)
        _para(fig,.07,.12,'Visualization only. It does not establish wound boundaries, infection or clinical cause.','PlusJakartaSans-Regular.ttf',8.5,100,.02)
        figs.append(fig)

    feats=list((scan.get('features') or {}).items())
    drift=list((scan.get('drift') or {}).items())
    lines=[('h','Colour measurements (raw image features, not predictions)')]+[('kv',(k,f'{v:.3f}')) for k,v in feats]
    if not feats:lines.append(('p','No valid measurements available.'))
    lines+=[('h',f"Colour comparison | previous scan: {scan.get('previous_id') or 'none linked'}")]+[('kv',(f'Delta {k}',f'{v:.3f}')) for k,v in drift]
    if not drift:lines.append(('p',scan.get('comparison_error') or 'No comparison available.'))
    lines+=[('h','Monitoring history (same wound)')]+[('kv',(s['scan_date'],f"{s['result']['risk_band']} | review: {s['review_status']}")) for s in history]
    fig=_page('Measurements and history','Colour features and monitoring record');y=.85
    for kind,val in lines:
        if y<.1:
            figs.append(fig);fig=_page('Measurements and history','Continued');y=.85
        if kind=='h':y=_section(fig,y-.008,val)
        elif kind=='p':y=_para(fig,.09,y,val,'PlusJakartaSans-Regular.ttf',9,95,.02)
        else:
            _text(fig,.09,y,val[0],'PlusJakartaSans-Regular.ttf',9);_text(fig,.5,y,val[1],'SpaceMono-Regular.ttf',9);y-=.02
    figs.append(fig)
    from matplotlib.lines import Line2D
    for i,f in enumerate(figs,1):
        f.add_artist(Line2D([.07,.93],[.062,.062],color='#C9D1DA',lw=.8,transform=f.transFigure))
        _text(f,.07,.04,'Wound Trace | Research screening tool. Not a diagnosis; clinician review required.','PlusJakartaSans-Regular.ttf',7.5,'#55657A')
        _text(f,.93,.04,f'{i} / {len(figs)}','SpaceMono-Regular.ttf',7.5,'#55657A',ha='right')
    return figs

@router.get('/scans/{identifier}/report.pdf')
def report(identifier:str):
    scan=required('scans',identifier)
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    out=io.BytesIO()
    history=sorted([s for s in store.records('scans') if s['patient_id']==scan['patient_id'] and enrich_scan(s)['wound_id']==enrich_scan(scan)['wound_id']],key=lambda s:(s['scan_date'],s['created_at']))
    image_path=store.ROOT/'images'/scan['image_file']
    try:
        with Image.open(image_path) as image: original=image.convert('RGB').copy()
    except OSError:raise HTTPException(404,'Stored photograph is unavailable')
    from .dashboard import guidance
    try:guide=guidance(identifier)
    except HTTPException:
        from woundwatch.guidance import build_guidance
        guide=build_guidance(scan['result'],None,scan.get('features'),None,outline_found=False)
    overlays=[]
    try:
        from .app import models_once
        from woundwatch.visualization import colour_heatmap,gradcam
        with pdf_lock:
            (_, _),(classifier,checkpoint)=models_once()
            overlays=[('CIELAB a* colour heatmap: measured red-green appearance',colour_heatmap(original))]
            try: overlays.append(('Elevated-logit Grad-CAM: model sensitivity visualization, not an explanation',gradcam(original,classifier,checkpoint)))
            except Exception: pass
    except Exception: overlays=[]
    with pdf_lock,PdfPages(out) as pdf:
        for fig in report_figures(scan,history,guide,original,overlays):pdf.savefig(fig)
    return Response(out.getvalue(),media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="woundtrace-{identifier[:8]}.pdf"'})
