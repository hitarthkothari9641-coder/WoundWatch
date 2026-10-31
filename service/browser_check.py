"""Patient/clinician browser checks. Requires a dedicated test-results data directory."""
import os
import sys
from pathlib import Path
from uuid import uuid4
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from service import store
from service.accounts import create_user

root=Path(__file__).resolve().parents[1]
if not os.getenv('WOUNDWATCH_DATA_DIR'):raise RuntimeError('Set WOUNDWATCH_DATA_DIR to the isolated server test directory')
if not store.ROOT.resolve().is_relative_to((root/'test-results').resolve()):raise RuntimeError('Browser tests must not write to real patient records')
url=os.getenv('WOUNDWATCH_TEST_URL','http://127.0.0.1:8001')
email='browser-'+uuid4().hex[:10]+'@example.org';password='browser-only-test-passphrase-123'
sample=str(root/'data/raw/surgwound/images/train/02b44effbbf2.jpg')
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1050})
    page=context.new_page();page.set_default_timeout(20000);errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:errors.append(f'{r.status}: {r.url}') if r.status>=400 and not(r.status==401 and r.url.endswith('/api/auth/me')) else None)
    page.goto(url)
    page.locator('#sign-up-tab').click()
    page.locator('#auth-form [name=name]').fill('Maya Research')
    page.locator('#auth-form [name=email]').fill(email)
    page.locator('#auth-form [name=password]').fill(password)
    page.locator('#auth-form [type=submit]').click()
    page.get_by_role('heading',name='Welcome back, Maya!',exact=True).wait_for()
    assert page.locator('nav a').all_text_contents()==['Home','My Profile','Settings']
    page.screenshot(path='test-results/patient-home-empty.png',full_page=True)
    page.locator('nav a[href="#profile"]').click()
    page.locator('#add-wound').click()
    page.locator('#dialog-form [name=surgery_type]').fill('Test surgery')
    page.locator('#dialog-form [name=surgery_date]').fill('2026-01-01')
    page.locator('#dialog-form [name=wound_location]').fill('Left knee')
    page.locator('#dialog-form [type=submit]').click();page.locator('#dialog').wait_for(state='hidden')
    for date_value in ('2026-01-02','2026-01-03'):
        page.locator('a[href="#scans"]').first.click()
        page.locator('#scan-form [name=scan_date]').fill(date_value)
        page.locator('#scan-form [name=photo_time]').fill('10:00')
        page.locator('#image-file').set_input_files(sample)
        if date_value.endswith('03'):
            page.locator('#scan-form [name=scan_date]').dispatch_event('change')
            page.locator('#upload-previous').select_option(index=1)
        with page.expect_response(lambda r:r.url.endswith('/api/scans') and r.request.method=='POST',timeout=60000) as response:
            page.locator('#analyse-button').click()
        assert response.value.status==201,response.value.text()
        page.locator('#scan-detail').wait_for()
    page.locator('nav a[href="#home"]').click()
    page.screenshot(path='test-results/patient-home-desktop.png',full_page=True)
    assert page.locator('aside,#sidebar,#menu').count()==0
    page.locator('#notification-bell').click()
    page.locator('#notification-popover').wait_for(state='visible')
    assert page.locator('#notification-popover .is-unread').count()==2
    page.locator('#mark-read').click()
    page.locator("#notification-popover .is-read").nth(1).wait_for()
    page.keyboard.press('Escape')
    assert page.locator('#notification-popover').is_hidden()
    assert page.locator('#notification-bell').get_attribute('aria-expanded')=='false'
    page.locator('nav a[href="#settings"]').click()
    page.get_by_role('heading',name='Settings',exact=True).wait_for()
    page.locator('nav a[href="#home"]').click()
    page.locator('.scan-card [data-open-scan]').first.click()
    page.locator('#scan-detail').wait_for()
    page.locator('[data-tab=colour]').click()
    page.locator('#colour-previous').select_option(index=1)
    with page.expect_response(lambda r:'/comparison' in r.url and r.request.method=='PATCH') as comparison:
        page.locator('#save-comparison').click()
    assert comparison.value.status==200
    page.get_by_text('Your comparison is saved.',exact=True).wait_for()
    page.locator('.visual-details summary').click()
    for kind in ('colour','gradcam'):
        page.locator(f'[data-visual={kind}]').click()
        page.locator(f'[data-visual={kind}]').locator('..').locator('.visual-output img').wait_for(timeout=60000)
    page.screenshot(path='test-results/patient-comparison.png',full_page=True)
    page.locator('[data-tab=report]').click()
    with page.expect_download() as downloaded:
        page.get_by_role('link',name='Download PDF report').click()
    downloaded.value.save_as('test-results/patient-report.pdf')
    assert Path('test-results/patient-report.pdf').read_bytes().startswith(b'%PDF')
    me=page.request.get(url+'/api/auth/me').json();patient_id=me['user']['patient_id']
    clinician_email='doctor-'+uuid4().hex[:10]+'@example.org'
    doctor=create_user(clinician_email,'Dr Research',password,'clinician')
    with store.connection() as db:db.execute('INSERT INTO assignments VALUES(?,?)',(doctor['id'],patient_id))
    page.locator('nav a[href="#home"]').click()
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Mobile page overflows'
    page.screenshot(path='test-results/patient-home-mobile.png',full_page=True)
    page.locator('nav a[href="#profile"]').click()
    page.get_by_role('heading',name='My Profile',exact=True).wait_for()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Mobile profile overflows'
    for width in (320,375,768,1024,1440):
        page.set_viewport_size({'width':width,'height':900})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'Overflow at {width}'
        assert page.locator('nav a').all_text_contents()==['Home','My Profile','Settings']
        assert page.locator('nav').is_visible()
    # Separate browser context proves that role choice follows server identity.
    doctor_context=browser.new_context(viewport={'width':1440,'height':1000});doc=doctor_context.new_page()
    doc.on('pageerror',lambda e:errors.append(str(e)));doc.goto(url)
    doc.locator('#auth-form [name=email]').fill(clinician_email);doc.locator('#auth-form [name=password]').fill(password)
    doc.locator('#auth-form [type=submit]').click()
    doc.get_by_role('heading',name='Welcome, Dr Research',exact=True).wait_for()
    doc.locator('[data-patient]').first.click()
    doc.get_by_role('heading',name='Maya Research',exact=True).wait_for()
    doc.screenshot(path='test-results/clinician-profile.png',full_page=True)
    assert doc.locator('nav a').all_text_contents()==['Home','My Profile','Settings']
    doc.locator('nav a[href="#profile"]').click()
    doc.get_by_role('heading',name='My Profile',exact=True).wait_for()
    assert doc.locator('.account-profile').inner_text().find('Dr Research')>=0
    assert doc.locator('#wound-journey').count()==0
    doc.locator('nav a[href="#home"]').click()
    doc.locator('#patient-search').fill('no matching patient')
    assert doc.locator('[data-patient]').count()==0
    doc.locator('#patient-search').fill('Maya')
    assert doc.locator('[data-patient]').count()==1
    page.set_viewport_size({'width':1440,'height':1050});page.locator('#logout').click()
    page.get_by_role('heading',name='Good to see you',exact=True).wait_for()
    assert page.request.get(url+'/api/patients').status==401
    assert not errors,errors
    print('PASS: patient registration/login, wound setup, two actual scans, per-wound timeline, both overlays, PDF download, mobile navigation, assigned clinician view and logout')
    browser.close()
