"""Record the patient dashboard flow at iPhone 18 Pro Max CSS size (440x956) against a LOCAL demo backend.

    DEMO_URL=http://127.0.0.1:8001 DEMO_LOGIN=dist/demo_local_login.txt python scripts/record_dashboard_demo.py

Writes dist/demo/dashboard.webm. Local synthetic account only; never point this at real patient data.
"""
import os
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = os.getenv('DEMO_URL', 'http://127.0.0.1:8001')
creds = dict(l.split(': ', 1) for l in Path(os.getenv('DEMO_LOGIN', root / 'dist/demo_local_login.txt')).read_text().splitlines() if ': ' in l)
sample = str(root / 'data/raw/surgwound/images/train/02b44effbbf2.jpg')
out = root / 'dist/demo'; shutil.rmtree(out / 'tmp', ignore_errors=True); out.mkdir(parents=True, exist_ok=True)

CAPTION = """(t)=>{let b=document.getElementById('demo-cap');if(!b){b=document.createElement('div');b.id='demo-cap';
b.style.cssText='position:fixed;left:12px;right:12px;top:10px;z-index:9999;background:#0C1E36;color:#fff;padding:10px 14px;border-radius:12px;font:600 15px/1.35 Plus Jakarta Sans,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.3);border-left:5px solid #D92638;pointer-events:none';document.body.append(b)}b.textContent=t}"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    ctx = browser.new_context(viewport={'width': 440, 'height': 956}, device_scale_factor=3, is_mobile=True, has_touch=True,
                              record_video_dir=str(out / 'tmp'), record_video_size={'width': 880, 'height': 1912})
    page = ctx.new_page(); page.set_default_timeout(30000)
    cap = lambda t, ms=2200: (page.evaluate(CAPTION, t), page.wait_for_timeout(ms))
    page.goto(url); page.wait_for_selector('#auth-form')
    cap('Wound Trace: a little care, a clearer picture.', 2500)
    page.locator('#assist-btn').click(); page.wait_for_timeout(800)
    cap('Built-in help in English, Hindi and Gujarati, with read-aloud.', 1800)
    page.get_by_role('button', name='हिन्दी').click(); page.wait_for_timeout(2200)
    page.get_by_role('button', name='ગુજરાતી').click(); page.wait_for_timeout(2000)
    page.get_by_role('button', name='English').click(); page.wait_for_timeout(600)
    page.locator('.assist-close').click(); page.wait_for_timeout(500)
    cap('Patients sign in with email and password.', 1500)
    page.locator('#auth-form [name=email]').fill(creds['email'])
    page.locator('#auth-form [name=password]').fill(creds['password'])
    page.locator('#auth-form [type=submit]').click()
    page.locator('nav a[href="#profile"]').wait_for(); cap('Home: one clear action, scan your wound.', 2500)
    page.locator('nav a[href="#profile"]').click(); page.locator('#add-wound').click()
    page.locator('#dialog-form [name=surgery_type]').fill('Knee surgery')
    page.locator('#dialog-form [name=surgery_date]').fill('2026-10-20')
    page.locator('#dialog-form [name=wound_location]').fill('Left knee')
    cap('Set up the wound once: surgery, date, location.', 2000)
    page.locator('#dialog-form [type=submit]').click(); page.locator('#dialog').wait_for(state='hidden')
    page.locator('a[href="#scans"]').first.click()
    page.locator('#scan-form [name=scan_date]').fill('2026-10-28'); page.locator('#scan-form [name=photo_time]').fill('10:00')
    page.locator('#image-file').set_input_files(sample)
    cap('Add a photo. Analysis runs on the Wound Trace model.', 2200)
    with page.expect_response(lambda r: r.url.endswith('/api/scans') and r.request.method == 'POST', timeout=90000):
        page.locator('#analyse-button').click()
    page.locator('#scan-detail').wait_for(); cap('Result for your care team: a screening signal, never a diagnosis.', 3000)
    page.mouse.wheel(0, 700); page.wait_for_timeout(1500)
    cap('Plain-language guidance: what to check, and what it cannot tell you.', 3000)
    page.locator('[data-tab=colour]').click(); page.wait_for_timeout(800)
    page.locator('.visual-details summary').click()
    for kind, text in (('colour', 'Colour map: where the skin looks redder.'), ('gradcam', 'Grad-CAM: where the model looked.')):
        page.locator(f'[data-visual={kind}]').click()
        page.locator(f'[data-visual={kind}]').locator('..').locator('.visual-output img').wait_for(timeout=90000)
        cap(text, 2800)
    page.locator('[data-tab=report]').click(); cap('One-tap branded PDF report for the doctor.', 2800)
    page.goto(url + '/#model'); page.wait_for_timeout(1500)
    cap('Honest model page: accuracy, limits and how it was tested.', 3500)
    page.mouse.wheel(0, 900); page.wait_for_timeout(2500)
    ctx.close(); browser.close()

vid = next((out / 'tmp').glob('*.webm')); shutil.move(vid, out / 'dashboard.webm'); shutil.rmtree(out / 'tmp')
print('wrote', out / 'dashboard.webm')
