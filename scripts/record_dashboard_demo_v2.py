"""Record patient (iPhone 18 Pro Max size) and doctor (desktop) dashboard views with several photos, on a LOCAL demo server.

    DEMO_URL=http://127.0.0.1:8002 WOUNDWATCH_DATA_DIR=dist/demo-data2 python scripts/record_dashboard_demo_v2.py

Creates two synthetic test accounts (passwords written to dist/demo_local_login2.txt), assigns the doctor to the patient,
then records dist/demo/patient_mobile.webm and dist/demo/doctor_desktop.webm. Never point at real patient data.
"""
import os
import secrets
import shutil
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'python'))
from service import store  # noqa: E402
from service.accounts import create_user  # noqa: E402

url = os.getenv('DEMO_URL', 'http://127.0.0.1:8002')
if not os.getenv('WOUNDWATCH_DATA_DIR'): raise SystemExit('Set WOUNDWATCH_DATA_DIR to the demo server data dir')
out = root / 'dist/demo'; out.mkdir(parents=True, exist_ok=True)

pw_p, pw_d = secrets.token_urlsafe(12) + 'Aa1', secrets.token_urlsafe(12) + 'Aa1'
patient = create_user('meera.test@example.org', 'Meera Test', pw_p, 'patient', None)
doctor = create_user('dr.rao.test@example.org', 'Dr Rao', pw_d, 'clinician')
with store.connection() as db: db.execute('INSERT INTO assignments VALUES(?,?)', (doctor['id'], patient['patient_id']))
(root / 'dist/demo_local_login2.txt').write_text(
    f"LOCAL DEMO ONLY ({url})\npatient: meera.test@example.org / {pw_p}\ndoctor: dr.rao.test@example.org / {pw_d}\n")
os.chmod(root / 'dist/demo_local_login2.txt', 0o600)

idx = pd.read_csv(root / 'results/inference/index.csv'); idx = idx[idx.status == 'ok']
pick = [idx[idx.risk_band == b].sort_values('probability').iloc[k] for b, k in (('low', 5), ('low', 40), ('elevated', 10), ('elevated', 60))]
photos = [(str(root / f'data/raw/surgwound/images/{r.split}/{r.id}.jpg'), d) for r, d in zip(pick, ('2026-10-22', '2026-10-25', '2026-10-28', '2026-10-31'))]

CAPTION = """(t)=>{let b=document.getElementById('demo-cap');if(!b){b=document.createElement('div');b.id='demo-cap';
b.style.cssText='position:fixed;left:12px;right:12px;top:10px;z-index:9999;background:#0C1E36;color:#fff;padding:10px 14px;border-radius:12px;font:600 15px/1.35 Plus Jakarta Sans,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.3);border-left:5px solid #D92638;pointer-events:none';document.body.append(b)}b.textContent=t}"""


def record(p, name, viewport, video, mobile, flow):
    tmp = out / f'tmp_{name}'; shutil.rmtree(tmp, ignore_errors=True)
    browser = p.chromium.launch(channel='chrome', headless=True)
    ctx = browser.new_context(viewport=viewport, device_scale_factor=3 if mobile else 1, is_mobile=mobile, has_touch=mobile,
                              record_video_dir=str(tmp), record_video_size=video)
    page = ctx.new_page(); page.set_default_timeout(60000)
    cap = lambda t, ms=2200: (page.evaluate(CAPTION, t), page.wait_for_timeout(ms))
    flow(page, cap); ctx.close(); browser.close()
    shutil.move(next(tmp.glob('*.webm')), out / f'{name}.webm'); shutil.rmtree(tmp); print('wrote', out / f'{name}.webm')


def patient_flow(page, cap):
    page.goto(url); page.wait_for_selector('#auth-form')
    cap('Patient view on iPhone 18 Pro Max screen size.', 2200)
    page.locator('#assist-btn').click(); page.wait_for_timeout(600)
    page.get_by_role('button', name='हिन्दी').click(); cap('Help in Hindi, Gujarati and English, with read-aloud.', 2600)
    page.get_by_role('button', name='English').click(); page.locator('.assist-close').click()
    page.locator('#auth-form [name=email]').fill('meera.test@example.org')
    page.locator('#auth-form [name=password]').fill(pw_p); page.locator('#auth-form [type=submit]').click()
    page.locator('nav a[href="#profile"]').wait_for(); cap('Signed in as a test patient.', 1800)
    page.locator('nav a[href="#profile"]').click(); page.locator('#add-wound').click()
    page.locator('#dialog-form [name=surgery_type]').fill('Abdominal surgery')
    page.locator('#dialog-form [name=surgery_date]').fill('2026-10-20')
    page.locator('#dialog-form [name=wound_location]').fill('Lower abdomen')
    cap('One-time wound setup.', 1800); page.locator('#dialog-form [type=submit]').click(); page.locator('#dialog').wait_for(state='hidden')
    for n, (photo, day) in enumerate(photos, 1):
        page.locator('a[href="#scans"]').first.click()
        page.locator('#scan-form [name=scan_date]').fill(day); page.locator('#scan-form [name=photo_time]').fill('09:30')
        page.locator('#image-file').set_input_files(photo)
        if n > 1:
            page.locator('#scan-form [name=scan_date]').dispatch_event('change'); page.locator('#upload-previous').select_option(index=1)
        cap(f'Photo {n} of {len(photos)}: {day}. Compared with the earlier photo.' if n > 1 else f'Photo {n} of {len(photos)}: {day}.', 1800)
        with page.expect_response(lambda r: r.url.endswith('/api/scans') and r.request.method == 'POST', timeout=120000):
            page.locator('#analyse-button').click()
        page.locator('#scan-detail').wait_for(); cap(f'Result {n}: a screening signal for the care team, not a diagnosis.', 2600)
        if n in (2, 4):
            page.mouse.wheel(0, 700); cap('Plain-language guidance and limits.', 2400)
            page.locator('[data-tab=colour]').click(); page.locator('.visual-details summary').click()
            for kind, text in (('colour', 'Colour map: where skin looks redder.'), ('gradcam', 'Grad-CAM: where the model looked.')):
                page.locator(f'[data-visual={kind}]').click()
                page.locator(f'[data-visual={kind}]').locator('..').locator('.visual-output img').wait_for(timeout=120000); cap(text, 2600)
    page.locator('[data-tab=report]').click(); cap('Branded PDF report for the doctor.', 2600)
    page.locator('nav a[href="#home"]').click(); cap('Home: timeline of all four photos.', 3000)
    page.locator('#logout').click()


def doctor_flow(page, cap):
    page.goto(url); page.wait_for_selector('#auth-form')
    cap('Doctor view on a desktop screen.', 2200)
    page.locator('#doctor-login-tab').click()
    page.locator('#auth-form [name=email]').fill('dr.rao.test@example.org'); page.locator('#auth-form [name=password]').fill(pw_d)
    page.locator('#auth-form [type=submit]').click(); page.get_by_role('heading', name='Welcome, Dr Rao', exact=True).wait_for()
    cap('Doctor dashboard: all assigned patients and their screening results.', 3200)
    page.locator('#patient-risk').select_option('elevated'); cap('Filter to Elevated cases first.', 2400)
    page.locator('#patient-risk').select_option('')
    page.locator('[data-patient]').first.click(); page.get_by_role('heading', name='Meera Test', exact=True).wait_for()
    cap('Patient record with every photograph and its trend.', 3000)
    page.locator('nav a[href="#home"]').click(); page.locator('.scan-card [data-open-scan]').first.click()
    page.locator('#doctor-review-form').wait_for(); cap('Open a scan, read the guidance, record a human review.', 3000)
    page.locator('#doctor-review-form [name=outcome]').select_option('follow_up')
    page.locator('#doctor-review-form [name=notes]').fill('Reviewed with patient. Repeat photo in 48 hours; examine in clinic if redness spreads.')
    cap('Clinician notes stay separate from the AI result.', 2200); page.locator('#doctor-review-form [type=submit]').click()
    page.get_by_text('Human review saved. AI screening was not changed.', exact=True).wait_for(); page.wait_for_timeout(2200)
    page.goto(url + '/#model'); cap('Model page: honest accuracy, confusion matrix and limits.', 4500)
    page.mouse.wheel(0, 900); page.wait_for_timeout(2500)


with sync_playwright() as p:
    record(p, 'patient_mobile', {'width': 440, 'height': 956}, {'width': 880, 'height': 1912}, True, patient_flow)
    record(p, 'doctor_desktop', {'width': 1280, 'height': 800}, {'width': 1280, 'height': 800}, False, doctor_flow)
