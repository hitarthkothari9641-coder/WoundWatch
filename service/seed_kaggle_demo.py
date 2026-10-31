"""Create an isolated Kaggle wound-category workflow demo with synthetic accounts."""
import csv
import hashlib
import json
import os
import secrets
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from woundwatch.config import CFG
from woundwatch.common import read_rgb
from woundwatch.kaggle_transfer import kaggle_originals


def main():
    root = CFG.root / 'runtime' / 'kaggle-demo-ready'
    db_path = root / 'woundwatch.sqlite'
    if db_path.exists():
        raise SystemExit(f'Demo database already exists: {db_path}. Refusing to overwrite accounts or reports.')
    os.environ['WOUNDWATCH_DATA_DIR'] = str(root)
    # Import after setting the isolated runtime path.
    from service import store
    from service.accounts import create_user
    from service.monitoring import wound_key

    paths, _, classes, _, _ = kaggle_originals(CFG.kaggle)
    class_to_paths = {name: [] for name in classes}
    for path in paths:
        category = path.parent.name
        class_to_paths[category].append(path)
    if len(classes) != 10 or any(not class_to_paths[name] for name in classes):
        raise SystemExit('Expected ten Kaggle class folders, each with at least one original image.')

    local_appdata = os.environ.get('LOCALAPPDATA')
    if not local_appdata:
        raise SystemExit('LOCALAPPDATA is not set; cannot save demo credentials outside the project.')
    credential_file = Path(local_appdata) / 'WoundWatch' / 'kaggle-demo-accounts.csv'
    credential_file.parent.mkdir(parents=True, exist_ok=True)

    accounts = []
    clinicians = []
    for i in range(1, 3):
        name = f'DEMO Doctor {i:02d}'
        email = f'demo.doctor{i:02d}@woundwatch.invalid'
        password = secrets.token_urlsafe(24)
        user = create_user(email, name, password, role='clinician')
        clinicians.append(user)
        accounts.append({'role': 'clinician', 'email': email, 'password': password, 'display_name': name})

    today = date.today()
    scans_for_report = []
    for index, category in enumerate(classes, start=1):
        name = f'DEMO Kaggle {category}'
        slug = ''.join(c.lower() if c.isalnum() else '.' for c in category).strip('.')
        email = f'demo.patient{index:02d}.{slug}@woundwatch.invalid'
        password = secrets.token_urlsafe(24)
        user = create_user(email, name, password, role='patient')
        accounts.append({'role': 'patient', 'email': email, 'password': password, 'display_name': name})
        patient_id = user['patient_id']
        surgery_date = (today - timedelta(days=30)).isoformat()
        location = f'DEMO Kaggle wound type: {category}'
        patient = store.get('patients', patient_id)
        patient.update(surgery_type='SYNTHETIC DEMO METADATA — not patient history',
                       surgery_date=surgery_date, wound_location=location,
                       demo_only=True, demo_source='Kaggle wound category')
        store.save('patients', patient)
        wound_id = wound_key(patient_id, surgery_date, location)
        wound = dict(id=wound_id, patient_id=patient_id, surgery_date=surgery_date,
                     wound_location=location, surgery_type=patient['surgery_type'],
                     created_at=datetime.now(timezone.utc).isoformat(), demo_only=True)
        with store.connection() as db:
            db.execute('INSERT INTO wounds VALUES(?,?,?)',
                       (wound_id, patient_id, json.dumps(wound)))
            for doctor in clinicians:
                db.execute('INSERT INTO assignments VALUES(?,?)', (doctor['id'], patient_id))

        source = sorted(class_to_paths[category], key=lambda p: p.name.casefold())[0]
        image = read_rgb(source)
        scan_id = uuid4().hex
        image_file = scan_id + '.png'
        image_path = root / 'images' / image_file
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, format='PNG')
        feature_row = None
        feature_error = None
        try:
            from woundwatch.visualization import colour_analysis
            from service.app import models_once
            (segmenter, _), _ = models_once()
            colour = colour_analysis(image, segmenter)
            feature_row, feature_error = colour['features'], colour['feature_error']
        except Exception as exc:
            colour = {'colour_method': 'unavailable', 'colour_size': list(image.size),
                      'ring_pixels': 0, 'colour_scope': 'No validated demo segmentation available.'}
            feature_error = type(exc).__name__
        result = {'status': 'defer', 'risk_band': 'defer', 'probability': None,
                  'reasons': [f'DEMO ONLY: Kaggle source category is {category}. These labels describe wound type, not infection outcome. Infection-risk prediction withheld.']}
        scan = dict(id=scan_id, patient_id=patient_id,
                    patient_snapshot={**patient, 'name': name}, scan_date=today.isoformat(),
                    surgery_date=surgery_date, wound_location=location, previous_id=None,
                    created_at=datetime.now(timezone.utc).isoformat(), result=result,
                    features=feature_row, feature_error=feature_error,
                    image_file=image_file, review_status='pending', demo_only=True,
                    source_dataset='Kaggle Wound Dataset', source_class=category,
                    source_image_name=source.name, wound_id=wound_id,
                    colour_method=colour['colour_method'], colour_size=colour['colour_size'],
                    ring_pixels=colour['ring_pixels'], colour_scope=colour['colour_scope'],
                    stages={'source_label': 'wound type only',
                                     'infection_classification': 'withheld',
                                     'longitudinal_comparison': 'not applicable: one unrelated category example'},
                    comparison_error='No previous scan: this is one category example, not a longitudinal patient visit.')
        with store.connection() as db:
            db.execute('INSERT INTO scans VALUES(?,?,?)',
                       (scan_id, patient_id, json.dumps(scan, allow_nan=False)))
            event = dict(id=uuid4().hex, scan_id=scan_id,
                         created_at=datetime.now(timezone.utc).isoformat(),
                         message='DEMO AI report awaiting healthcare review · infection prediction withheld',
                         read=False)
            db.execute('INSERT INTO events VALUES(?,?)', (event['id'], json.dumps(event)))
            db.commit()
        scans_for_report.append((category, source, image, feature_row, scan_id))

    with credential_file.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['role', 'email', 'password', 'display_name'])
        writer.writeheader()
        writer.writerows(accounts)

    reports = root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    make_comparison_report(scans_for_report, reports)
    print('Created isolated Kaggle wound-type demo: 10 synthetic patient accounts, 2 synthetic clinician accounts, 10 pending reports.')
    print(f'Credentials stored locally at: {credential_file}')
    print(f'Category comparison PDF: {reports / "kaggle_category_comparison.pdf"}')
    print(f'Runtime database and images: {root}')
    print('No Kaggle class is treated as an infection label. One unrelated example per category; no longitudinal claims.')


def make_comparison_report(rows, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    report_rows = []
    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    for index, (category, source, image, features, scan_id) in enumerate(rows):
        row, col = divmod(index, 2)
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(f'{category}\nOne sample; not infection ground truth', fontsize=9)
        ax.axis('off')
        try:
            from woundwatch.visualization import colour_heatmap
            heatmap = colour_heatmap(image)
            axes[row, col + 2].imshow(heatmap)
            axes[row, col + 2].set_title('CIELAB a* appearance heatmap', fontsize=9)
            axes[row, col + 2].axis('off')
        except Exception:
            axes[row, col + 2].text(.1, .5, 'Heatmap unavailable')
            axes[row, col + 2].axis('off')
        report_row = {'category': category, 'source_image': source.name,
                      'infection_prediction': 'WITHHELD', 'longitudinal_comparison': 'not applicable'}
        for key, value in (features or {}).items():
            report_row[key] = value
        report_rows.append(report_row)
    fig.suptitle('WoundWatch demo · Kaggle wound category examples\nNot patient records; categories are not infection labels', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out_dir / 'kaggle_category_comparison.png', dpi=130)
    plt.close(fig)
    pd.DataFrame(report_rows).to_csv(out_dir / 'kaggle_category_analysis.csv', index=False)

    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from PIL import Image
    with PdfPages(out_dir / 'kaggle_category_comparison.pdf') as pdf:
        fig, axes = plt.subplots(5, 4, figsize=(11.7, 16.5))
        for index, (category, source, image, features, scan_id) in enumerate(rows):
            row, col = divmod(index, 2)
            axes[row, col].imshow(image); axes[row, col].set_title(category); axes[row, col].axis('off')
            try: axes[row, col + 2].imshow(__import__('woundwatch.visualization', fromlist=['colour_heatmap']).colour_heatmap(image))
            except Exception: pass
            axes[row, col + 2].set_title('CIELAB a* heatmap'); axes[row, col + 2].axis('off')
        fig.suptitle('Kaggle wound-category examples and colour heatmaps\nSynthetic demo only · no infection prediction · no longitudinal patient comparison')
        fig.tight_layout(rect=(0, 0, 1, .96)); pdf.savefig(fig); plt.close(fig)
        fig=Figure(figsize=(8.27,11.69))
        text=('Analysis and limitations\n\n'
              'These ten images are single examples from ten wound-type folders. They are not ten verified patients, '
              'and the source categories do not label infection. No infection probability is computed or shown.\n\n'
              'Colour maps show image a* appearance only. They are affected by lighting, white balance, skin tone, '
              'image processing and segmentation errors. Grad-CAM is a model sensitivity visualization, not an explanation.\n\n'
              'The demo accounts and clinical metadata are synthetic. Each doctor account has access to the ten demo '
              'reports, all marked pending. Recording a human review is not diagnostic approval.\n\n'
              'Kaggle source license is unverified/unknown in the local project notes. Keep examples local until use rights are confirmed.')
        fig.text(.08,.92,text,fontsize=12,va='top',wrap=True,parse_math=False);pdf.savefig(fig)


if __name__ == '__main__':
    main()
