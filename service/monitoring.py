"""Patient profiles, wound identities and private scan reports."""
import io
import json
import threading
import textwrap
from datetime import date
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

@router.get('/scans/{identifier}/report.pdf')
def report(identifier:str):
    scan=required('scans',identifier)
    # Reuse installed Matplotlib's PDF backend; no browser or new dependency.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    import numpy as np
    out=io.BytesIO()
    history=sorted([s for s in store.records('scans') if s['patient_id']==scan['patient_id'] and enrich_scan(s)['wound_id']==enrich_scan(scan)['wound_id']],key=lambda s:(s['scan_date'],s['created_at']))
    snapshot=scan['patient_snapshot']
    demo=bool(scan.get('demo_only'))
    lines=[f"Patient: {snapshot['name']} ({scan['patient_id']})",f"Surgery: {snapshot.get('surgery_type','Not recorded')} | {scan['surgery_date']}",
           f"Wound: {scan['wound_location']}",f"Photograph date: {scan['scan_date']}",f"Analysis time: {scan['created_at']}",
           f"Screening: {scan['result']['risk_band']} | Workflow: {scan['result']['status']}",
           'Elevated-label probability: '+(f"{100*scan['result']['probability']:.1f}%" if scan['result']['probability'] is not None else 'Unavailable'),
           f"Healthcare review: {scan['review_status']} (recording review is not diagnosis approval)",*scan['result']['reasons']]
    if demo:
        lines += ['DEMO DATA ONLY: synthetic account and Kaggle wound-type example.',
                  f"Source category: {scan.get('source_class','unknown')}. Wound-type category is not an infection outcome.",
                  'Infection-risk prediction is withheld. This example must not be used for clinical care.']
    else:
        lines += ['Analysis: the reported model output is an experimental screening result for clinician review.',
                  'The probability is a model estimate for the training label, not diagnostic certainty.',
                  'Colour measurements describe image appearance only; lighting, skin tone and segmentation affect them.']
    image_path=store.ROOT/'images'/scan['image_file']
    try:
        with Image.open(image_path) as image: original=image.convert('RGB').copy()
    except OSError:raise HTTPException(404,'Stored photograph is unavailable')
    overlays=[]
    try:
        from .app import models_once
        from woundwatch.visualization import colour_heatmap,gradcam
        with pdf_lock:
            (_, _),(classifier,checkpoint)=models_once()
            overlays=[('CIELAB a* colour heatmap',colour_heatmap(original))]
            try: overlays.append(('Elevated-logit Grad-CAM (model sensitivity visualization, not an explanation)',gradcam(original,classifier,checkpoint)))
            except Exception: pass
    except Exception: overlays=[]
    try:
        with pdf_lock,PdfPages(out) as pdf:
            fig=Figure(figsize=(8.27,11.69));fig.text(.08,.94,'Wound Trace | Monitoring report',fontsize=20,color='#0C1E36',parse_math=False)
            y=.89
            for line in lines:
                for segment in textwrap.wrap(line,90):fig.text(.08,y,segment,fontsize=10,parse_math=False);y-=.025
            ax=fig.add_axes([.1,.17,.8,max(.12,y-.22)])
            ax.imshow(original)
            ax.axis('off');fig.text(.08,.1,'Research screening only. This report does not confirm or exclude infection.',fontsize=10,parse_math=False)
            fig.text(.08,.07,'This does not replace clinical examination or professional medical diagnosis.',fontsize=10,parse_math=False);pdf.savefig(fig)
            for title,overlay in overlays:
                fig=Figure(figsize=(8.27,11.69));fig.text(.08,.94,'Wound Trace | Visual analysis',fontsize=18,color='#0C1E36',parse_math=False)
                ax=fig.add_axes([.1,.18,.8,.7]);ax.imshow(overlay);ax.axis('off')
                fig.text(.08,.1,title,fontsize=10,parse_math=False)
                fig.text(.08,.07,'Visualization only. It does not establish wound boundaries, infection, or clinical cause.',fontsize=9,parse_math=False);pdf.savefig(fig)
            detail=['COLOUR MEASUREMENTS (raw image features, not predictions)']
            detail.extend([f'{k}: {v:.3f}' for k,v in (scan.get('features') or {}).items()] or ['No valid measurements available.'])
            detail+=['','ANALYSIS NOTES','CIELAB a* heatmap shows image redness appearance; it is not a diagnosis.',
                     'Grad-CAM visualizes model sensitivity and is not a causal explanation.',
                     'Differences can reflect lighting, white balance, skin tone, pose, or segmentation error.',
                     'A single image cannot establish healing change.','', 'COLOUR COMPARISON',f"Previous scan: {scan.get('previous_id') or 'None linked'}"]
            detail.extend([f'Delta {k}: {v:.3f}' for k,v in (scan.get('drift') or {}).items()] or [scan.get('comparison_error') or 'No comparison available.'])
            detail+=['','MONITORING HISTORY (same wound)']+[f"{s['scan_date']} | {s['result']['risk_band']} | review: {s['review_status']}" for s in history]
            wrapped=[part for line in detail for part in (textwrap.wrap(line,95) or [''])]
            for offset in range(0,len(wrapped),40):
                fig=Figure(figsize=(8.27,11.69));fig.text(.08,.94,'Wound Trace | Measurements and history',fontsize=17,color='#0C1E36',parse_math=False)
                for i,line in enumerate(wrapped[offset:offset+40]):fig.text(.08,.89-i*.019,line,fontsize=9,parse_math=False)
                pdf.savefig(fig)
    except OSError:raise HTTPException(404,'Stored photograph is unavailable')
    return Response(out.getvalue(),media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="woundwatch-{identifier[:8]}.pdf"'})
