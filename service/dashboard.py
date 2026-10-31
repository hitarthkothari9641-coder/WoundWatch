"""Patient and scan workflows built on the existing inference functions."""
import io
import json
import threading
from datetime import date, datetime, timezone
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal
from woundwatch.color_features import feature_row
from woundwatch.inference import predict_image, segment_wound
from woundwatch.config import CFG
from woundwatch.common import read_rgb
from woundwatch.visualization import colour_analysis, colour_heatmap, gradcam, compare_features
from . import store
from .accounts import actor, clinician, authorize, patient_ids, audit

router = APIRouter(prefix='/api')
inference_lock = threading.Lock()

def now():
    return datetime.now(timezone.utc).isoformat()

def required(table, identifier):
    value = store.get(table, identifier)
    if value is None:
        raise HTTPException(404, 'Record not found')
    if table=='patients':authorize(value['id'])
    elif table in ('scans','wounds'):authorize(value['patient_id'])
    return value

class Patient(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    name: str = Field(min_length=1, max_length=100)
    age: int | None = Field(default=None, ge=0, le=120)
    surgery_type: str = Field(min_length=1, max_length=100)
    surgery_date: date
    wound_location: str = Field(min_length=1, max_length=100)
    monitoring_status: Literal['monitoring', 'follow-up', 'discharged'] = 'monitoring'

    @field_validator('surgery_date')
    @classmethod
    def past_date(cls, value):
        if value > date.today():
            raise ValueError('Surgery date cannot be in the future')
        return value

@router.get('/patients')
def patients():
    allowed=patient_ids()
    sites=store.records('wounds')+store.records('scans')
    return [{**p,'surgery_dates':sorted({s['surgery_date'] for s in sites if s['patient_id']==p['id']}|({p['surgery_date']} if p.get('surgery_date') else set()))}
            for p in store.records('patients') if p['id'] in allowed]

@router.post('/patients', status_code=201)
def create_patient(body: Patient):
    user=clinician()
    record = dict(id='P-' + uuid4().hex[:10], created_at=now(), **body.model_dump(mode='json'))
    store.save('patients', record)
    with store.connection() as db:
        db.execute('INSERT INTO assignments VALUES(?,?)',(user['id'],record['id']))
        audit(db,'patient_created',record['id'])
    return record

@router.put('/patients/{identifier}')
def edit_patient(identifier: str, body: Patient):
    clinician()
    record = required('patients', identifier)
    record.update(body.model_dump(mode='json'))
    store.save('patients', record)
    return record

@router.get('/scans')
def scans(patient_id: str | None = None):
    from .monitoring import enrich_scan
    allowed=patient_ids()
    if patient_id:authorize(patient_id)
    return [enrich_scan(s) for s in store.records('scans') if s['patient_id'] in allowed and (not patient_id or s['patient_id'] == patient_id)]

@router.get('/scans/{identifier}/image')
def scan_image(identifier: str):
    scan = required('scans', identifier)
    return FileResponse(store.ROOT / 'images' / scan['image_file'], media_type='image/png')

@router.post('/scans', status_code=201)
def create_scan(file: UploadFile = File(...), patient_id: str = Form(...),
                scan_date: date = Form(...), wound_location: str = Form(...),
                surgery_date: date = Form(...), previous_id: str = Form(''), wound_id: str = Form(''),
                captured_at: str = Form('')):
    from .app import decode_image, models_once, MAX_BYTES
    patient = required('patients', patient_id)
    from .monitoring import wound_key, get_wound
    if wound_id:
        wound=get_wound(patient_id,wound_id)
        if wound['surgery_date']!=surgery_date.isoformat() or wound['wound_location'].casefold()!=wound_location.strip().casefold():
            raise HTTPException(422,'Selected wound does not match the supplied surgery and location')
        patient={**patient,'surgery_type':wound['surgery_type'],'surgery_date':wound['surgery_date'],'wound_location':wound['wound_location']}
    if captured_at:
        try:
            capture=datetime.fromisoformat(captured_at)
            if not capture.tzinfo or capture.date()!=scan_date or capture.timestamp()>datetime.now(timezone.utc).timestamp()+300:
                raise ValueError()
        except ValueError:raise HTTPException(422,'Enter a valid photograph time with timezone matching the scan date')
    location = wound_location.strip()
    if not location or len(location) > 100:
        raise HTTPException(422, 'Wound location must contain 1–100 characters')
    if scan_date > date.today() or scan_date < surgery_date:
        raise HTTPException(422, 'Scan must be on or after surgery and not in the future')
    previous = required('scans', previous_id) if previous_id else None
    if previous and (previous['patient_id'] != patient_id or previous['wound_location'].casefold() != location.casefold()
                     or previous['surgery_date'] != surgery_date.isoformat() or previous['scan_date'] > scan_date.isoformat()):
        raise HTTPException(422, 'Previous scan must belong to this patient, surgery and wound, and precede this scan')
    try:
        image = decode_image(file.file.read(MAX_BYTES + 1))
    finally:
        file.file.close()
    try:
        with inference_lock:
            (seg, _), (clf, checkpoint) = models_once()
            result = predict_image(image, seg, clf, checkpoint)
            colour = colour_analysis(image, seg)
            features, feature_error = colour['features'], colour['feature_error']
    except Exception:
        raise HTTPException(503, 'Inference failed; check model installation')
    from .app import model_fingerprints
    colour['colour_segmenter_sha256']=model_fingerprints()['segmenter']
    drift, comparison_error = compare_features(colour,previous) if previous else (None,'No previous scan linked. Select another scan of the same wound to calculate drift.')
    identifier = uuid4().hex
    scan = dict(id=identifier, patient_id=patient_id, patient_snapshot=patient, scan_date=scan_date.isoformat(),
                surgery_date=surgery_date.isoformat(), wound_location=location, previous_id=previous_id or None,
                created_at=now(), result=result, features=features, feature_error=feature_error, drift=drift,
                image_file=identifier + '.png', review_status='pending',
                model='classifier.pt / segmenter.pt',
                stages={'decode': 'complete', 'preprocessing': 'complete', 'segmentation': 'complete',
                        'quality_gate': 'rejected' if result['status'] == 'retake' else 'complete',
                        'classification': 'skipped' if result['status'] == 'retake' else 'complete',
                        'colour_features': 'complete' if features else 'unavailable' if feature_error else 'skipped',
                        'colour_comparison': 'complete' if drift else 'unavailable'})
    from .app import model_fingerprints, model_paths
    scan.update(colour, comparison_error=comparison_error, model_fingerprints=model_fingerprints(),
                wound_id=wound_key(patient_id,surgery_date.isoformat(),location),captured_at=captured_at or None,
                model=' / '.join(p.name for p in model_paths().values()),
                colour_warning='Quality gate requested a retake; raw colour is unreliable.' if result['status']=='retake' else None)
    with store.connection() as db:
        path = store.ROOT / 'images' / scan['image_file']
        try:
            image.save(path, format='PNG')  # Re-encode without EXIF or original filename.
            db.execute('INSERT INTO scans VALUES(?,?,?)', (identifier, patient_id, json.dumps(scan, allow_nan=False)))
            message = 'Scan stored · ' + result['status']
            if result['risk_band'] == 'elevated': message += ' · elevated screening result for clinical review'
            event = dict(id=uuid4().hex, scan_id=identifier, created_at=now(), message=message, read=False)
            db.execute('INSERT INTO events VALUES(?,?)', (event['id'], json.dumps(event)))
            audit(db,'scan_created',identifier)
            db.commit()
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return scan

class Comparison(BaseModel):
    previous_id: str | None = None

@router.post('/scans/{identifier}/reanalyse',status_code=201)
def reanalyse_scan(identifier:str):
    source=required('scans',identifier)
    try:handle=(store.ROOT/'images'/source['image_file']).open('rb')
    except OSError:raise HTTPException(404,'Saved photograph is missing. Please upload it again.')
    with handle:
        scan=create_scan(file=UploadFile(file=handle,filename='saved.png'),patient_id=source['patient_id'],
            scan_date=date.fromisoformat(source['scan_date']),wound_location=source['wound_location'],
            surgery_date=date.fromisoformat(source['surgery_date']),previous_id=source.get('previous_id') or '',
            wound_id='',captured_at=source.get('captured_at') or '')
    scan['reanalysis_of']=identifier
    with store.connection() as db:db.execute('UPDATE scans SET payload=? WHERE id=?',(json.dumps(scan),scan['id']))
    return scan

def validate_previous(scan,previous):
    if (previous['id']==scan['id'] or previous['patient_id']!=scan['patient_id']
        or previous['surgery_date']!=scan['surgery_date']
        or previous['wound_location'].casefold()!=scan['wound_location'].casefold()
        or (previous['scan_date'],previous['created_at']) >= (scan['scan_date'],scan['created_at'])):
        raise HTTPException(422,'Choose an earlier scan of the same patient, surgery and wound')

@router.patch('/scans/{identifier}/comparison')
def link_comparison(identifier: str,body: Comparison):
    scan=required('scans',identifier)
    previous=required('scans',body.previous_id) if body.previous_id else None
    if previous: validate_previous(scan,previous)
    scan['previous_id']=body.previous_id
    scan['drift'],scan['comparison_error']=compare_features(scan,previous) if previous else (None,'No previous scan linked.')
    scan['stages']['colour_comparison']='complete' if scan['drift'] else 'unavailable'
    with store.connection() as db:
        db.execute('UPDATE scans SET payload=? WHERE id=?',(json.dumps(scan),identifier))
    return scan

@router.post('/scans/{identifier}/colour')
def recalculate_colour(identifier: str):
    from .app import models_once
    scan=required('scans',identifier)
    try:
        with inference_lock:
            (segmenter,_),_=models_once()
            colour=colour_analysis(read_rgb(store.ROOT/'images'/scan['image_file']),segmenter)
    except Exception:
        raise HTTPException(503,'Colour analysis failed; check image storage and model installation')
    with store.connection() as db:
        # Reload after inference so a concurrent review/comparison is preserved.
        scan=json.loads(db.execute('SELECT payload FROM scans WHERE id=?',(identifier,)).fetchone()[0])
        from .app import model_fingerprints
        scan.update(colour,colour_updated_at=now(),colour_segmenter_sha256=model_fingerprints()['segmenter'],
                    colour_warning='Quality gate requested a retake; raw colour is unreliable.' if scan['result']['status']=='retake' else None)
        scan['stages']['colour_features']='complete' if colour['features'] else 'unavailable'
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM scans')]
        by_id={s['id']:s for s in rows};by_id[identifier]=scan
        for s in by_id.values():
            if s['id']==identifier or s.get('previous_id')==identifier:
                previous=by_id.get(s.get('previous_id'))
                s['drift'],s['comparison_error']=compare_features(s,previous) if previous else (None,'No previous scan linked.')
                s['stages']['colour_comparison']='complete' if s['drift'] else 'unavailable'
                db.execute('UPDATE scans SET payload=? WHERE id=?',(json.dumps(s),s['id']))
    return scan

@router.get('/scans/{identifier}/visualization/{kind}')
def visualization(identifier: str,kind: Literal['colour','gradcam']):
    from .app import models_once,model_fingerprints
    scan=required('scans',identifier)
    if kind=='gradcam' and scan['result']['status']=='retake':
        raise HTTPException(422,'Grad-CAM unavailable: classification was skipped by the quality gate')
    if kind=='gradcam' and scan.get('model_fingerprints',{}).get('classifier')!=model_fingerprints()['classifier']:
        raise HTTPException(409,'This older scan has no matching model version. Choose Re-analyse saved image, then generate Grad-CAM from the new analysis.')
    try:
        image=read_rgb(store.ROOT/'images'/scan['image_file'])
        if kind=='colour': output=colour_heatmap(image)
        else:
            with inference_lock:
                _,(classifier,checkpoint)=models_once()
                output=gradcam(image,classifier,checkpoint)
        buffer=io.BytesIO();output.save(buffer,format='PNG')
        return Response(buffer.getvalue(),media_type='image/png')
    except ValueError as exc:
        if kind=='gradcam' and str(exc).startswith('No positive'): raise HTTPException(422,str(exc))
        raise HTTPException(503,'Visualization unavailable; check stored image and model')
    except Exception:
        raise HTTPException(503,'Visualization unavailable; check stored image and model')

class Review(BaseModel):
    status: Literal['pending', 'reviewed']

@router.patch('/scans/{identifier}/review')
def review(identifier: str, body: Review):
    user=clinician()
    scan = required('scans', identifier)
    scan.update(review_status=body.status, reviewed_at=now() if body.status == 'reviewed' else None,reviewed_by=user['id'])
    with store.connection() as db:
        db.execute('UPDATE scans SET payload=? WHERE id=?', (json.dumps(scan), identifier))
        audit(db,'scan_reviewed',identifier)
    return scan

@router.get('/notifications')
def notifications():
    allowed={s['id'] for s in scans()}
    with store.connection() as db:
        read={r[0] for r in db.execute('SELECT event_id FROM event_reads WHERE user_id=?',(actor()['id'],))}
    return [{**e,'read':e['id'] in read} for e in store.records('events') if e.get('scan_id') in allowed]

@router.post('/notifications/read')
def mark_read():
    with store.connection() as db:
        for event in notifications():db.execute('INSERT OR IGNORE INTO event_reads VALUES(?,?)',(actor()['id'],event['id']))
    return {'status': 'ok'}

@router.get('/settings')
def settings():
    from .app import MAX_BYTES, MAX_PIXELS
    from .security import configuration
    production,username,_,_=configuration()
    return {'max_upload_bytes': MAX_BYTES, 'max_image_pixels': MAX_PIXELS,
            'mode': 'Local research workspace', 'lighting_correction': False,
            'classifier': 'ResNet-18 / binary Low versus Elevated',
            'segmenter': 'DeepLabV3+ / ResNet-18', 'authentication': True,
            'production_guards':production,'visualization_version':2,'account_version':1}
