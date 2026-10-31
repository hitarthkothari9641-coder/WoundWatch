# WoundWatch: patient and healthcare-worker guide

## Start the application

From `C:\Users\Param\WoundWatch` in PowerShell:

```powershell
.\start.ps1
```

Open **http://127.0.0.1:8000**. If an older server is running, stop it in its
terminal with **Ctrl+C** first. After restarting, refresh with **Ctrl+Shift+R**.
Do not start a second server on the same port.

For a fresh checkout, install dependencies first:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r service\requirements-dev.txt
.\start.ps1
```

If PowerShell blocks scripts, use:

```powershell
$env:PYTHONPATH = "$PWD\python"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '2'
.\.venv\Scripts\python -m uvicorn service.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

## New patient account

1. Choose **Create patient account**. Enter your name, email and a password of
   at least 12 characters. There is no preset/default password.
2. Open **My Profile → Add wound**. Enter your surgery type, surgery date and
   wound location. Add separate records for separate surgical sites.
3. Open **Scan My Wound**. Choose the wound, photograph date/time, and a clear
   JPG or PNG. Check the preview and choose **Submit for AI analysis**.
4. Your photograph and the actual model result are saved. A retake result means
   classification did not run; read the photo-quality notes.
5. **My Profile → My Wound Journey → View details** opens the original stored
   photograph, result, quality notes, colour comparison and report.

If you already have records in the old app, ask the local administrator to
create your account linked to that patient ID instead of registering another
empty record. No existing record is automatically exposed to a new account.

## Colour heatmap and Grad-CAM procedure

1. Sign in and open **My Profile → My Wound Journey**.
2. Select **View details** on a photograph.
3. In **Screening result** or **Colour comparison**, expand **Explore colour
   heatmap and AI attention map**.
4. Choose **Generate colour heatmap** or **Generate Grad-CAM**. Wait for the
   picture to appear below its button.

The colour heatmap needs only one image. It shows measured red–green colour
appearance, not infection. Grad-CAM shows coarse model attention, not infected
tissue or prediction confidence.

For an older analysis with no recorded model version, choose **Re-analyse saved
image** first. It creates a separate analysis using the saved photograph and
current models, preserving the original result. Grad-CAM remains unavailable if
classification is skipped due to a retake, or there is no positive activation.
The UI explains these cases; it does not fabricate an overlay.

If the page says the backend is out of date or a visualization route is missing,
stop the old server with Ctrl+C, run `start.ps1`, and refresh. Previously, the
browser could load updated JavaScript while an old Python process still lacked
the new API routes; restarting the server resolves that mismatch.

## Compare two photographs

1. Upload at least two photographs of the **same wound and surgery**.
2. Open the later scan in **My Wound Journey → View details → Colour comparison**.
3. Select its earlier photograph and click **Compare photographs**.
4. If the app says the measurement methods or model versions differ, open each
   scan and click **Recalculate colour**, then compare again.

A single image cannot show longitudinal colour drift. Empty segmentation or
insufficient measurements produce a warning. Photographs from different wounds
cannot be linked. Recalculation updates colour measurements, not the saved
screening prediction. The extraction/comparison algorithms remain unchanged by
the patient-interface redesign.

## Download a report

Open **My Profile → My Wound Journey → View details → Reports → Download PDF
report**. This downloads an actual PDF containing available patient/surgery
information, the photograph, screening output, colour features/comparison and
the same wound's history. Missing values are labelled rather than invented.
The browser print option is also available.

## Healthcare-worker accounts (local administrator)

Public registration creates **patient accounts only**. Clinician privileges
cannot be requested through the registration form or forged in an API request.
The administrator runs these commands on the trusted application host. Passwords
are prompted privately; never put a password in a command or commit it to Git.

Create a clinician:

```powershell
.\.venv\Scripts\python -m service.manage create-user --role clinician --email doctor@example.org --name "Dr Example"
```

Find existing patient IDs (this command prints local names; use privately):

```powershell
.\.venv\Scripts\python -m service.manage list-records
```

Create a patient login for an existing record, replacing `P-REPLACE`:

```powershell
.\.venv\Scripts\python -m service.manage create-user --role patient --patient-id P-REPLACE --email patient@example.org --name "Patient Name"
```

Assign a patient to a clinician:

```powershell
.\.venv\Scripts\python -m service.manage assign --email doctor@example.org --patient-id P-REPLACE
```

Revoke that assignment:

```powershell
.\.venv\Scripts\python -m service.manage unassign --email doctor@example.org --patient-id P-REPLACE
```

Clinicians sign in using **Doctor Login** on the shared sign-in page. Their role
comes from the database, not the login selector; a mismatch is rejected and
does not leave an active session. Both roles use **Home, My Profile, Settings**
in the navigation capsule, with notifications, account profile and sign-out in
a separate header capsule. The clinician Home screen contains summary counts,
a review queue, recent scans, and patient management with search, risk and
monitoring filters, and latest-scan sorting. Selecting an authorised patient
opens their complete profile/timeline. A clinician's **My Profile** shows their
own account details. Creating a new patient record assigns it to that clinician;
the administrator separately creates its patient login.

On a scan, clinicians can record one of the explicit human-review outcomes and
optional clinical notes, then mark the record reviewed. Entries include the
reviewer account and time and are retained as history. These are clinician
entries, not model output; they do not alter the AI screening result.

Use `WOUNDWATCH_DATA_DIR` consistently if your data directory is not the default
`runtime`. Administrator commands must target the same database as the server.

## Passwords and sessions

Change a password under **Settings → Change password**. This revokes all of
that account's sessions. Sign-out revokes the current session; sessions expire
after eight hours. Password recovery email and email-address verification are
not configured. After verifying the user's identity offline, the administrator
can reset a password:

```powershell
.\.venv\Scripts\python -m service.manage reset-password --email patient@example.org
```

## Retention and privacy

The existing SQLite database, patient records and stored images are retained.
New tables add users, sessions, assignments, wound records, event-read states,
authentication rate limits and an audit of selected changes. Legacy scan JSON
and image files are not destructively migrated. Wound IDs can be derived from
their existing patient, surgery date and location.

Unassigned legacy records are inaccessible to new accounts until the local
administrator explicitly links them. Data is not placed in a public directory.
Images, PDF reports, colour analysis, comparisons and model visualizations all
require both a signed-in account and patient authorization.

There is no automatic deletion. Records remain until an authorised administrator
applies the organisation's retention policy. Use encrypted storage/backups and
define consent, retention, recovery and deletion procedures before clinical use.

## Testing checklist

- Register two patients; each should see only their own profile, wounds and scans.
- Log in as an unassigned clinician; the patient list should be empty.
- Choose the wrong account type at login; verify no active session is retained.
- Assign one patient; only that patient's records become available.
- Try another patient's image/report URL: the server must deny it (404).
- Add two different wounds; their timelines and comparisons must remain separate.
- Upload a JPG/PNG; check preview, actual inference and historical storage.
- Test a blurry image; expect quality notes rather than an invented probability.
- Generate both maps for a quality-accepted scan; use saved-image reanalysis for
  older unversioned scans.
- Link a previous scan, recalculate colours, download and open the PDF.
- Record a clinician review outcome and notes; verify they remain distinct from
  the AI result and that unauthorized users cannot write reviews.
- Sign out, then try a private API URL: expect 401.
- Check both header capsules, compact scan/latest-update cards and removal of
  the Home photo-help card.
- Test narrow screens, keyboard navigation, notification popover and
  persistent mobile scan action.

Automated checks:

```powershell
$env:PYTHONPATH = "$PWD\python"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '2'
.\.venv\Scripts\python -m unittest discover -s python\tests -v
.\.venv\Scripts\python -m unittest service.test_dashboard service.test_security service.test_accounts -v
node --check service/web/app.js
node --check service/web/views.js
node --check service/web/visualizations.js
```

For browser checks, use a separate test database, never real patient data:

```powershell
$env:WOUNDWATCH_DATA_DIR = "$PWD\test-results\patient-browser-runtime"
$env:PYTHONPATH = "$PWD\python"
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python -m uvicorn service.app:app --host 127.0.0.1 --port 8001 --no-access-log
```

In a second terminal set the same `WOUNDWATCH_DATA_DIR`, then run
`.\.venv\Scripts\python service/browser_check.py`. This requires Playwright
and installed Chrome. The test creates isolated test accounts and records.

## Files and implementation

Created in this redesign:

- `service/accounts.py`: password hashing, session cookies, CSRF, login limits,
  registration and backend patient permissions.
- `service/manage.py`: trusted-host clinician provisioning, patient linking,
  assignment/revocation and password reset commands.
- `service/monitoring.py`: permitted profile edits, separate wounds and private PDF reports.
- `service/web/views.js`: patient/clinician dashboard, filters, timeline and
  human-review interface.
- `service/test_accounts.py`: isolation, sessions, roles, wounds, legacy image
  reanalysis and PDF tests.
- `PATIENT_GUIDE.md`: this walkthrough and file inventory.

Modified:

- `service/app.py`, `dashboard.py`, `security.py`, `store.py`: authenticated
  routes, ownership checks, additive storage, per-user notifications,
  reanalysis of saved images and role-restricted healthcare reviews.
- `service/web/index.html`, `app.js`, `style.css`, `visualizations.js`: dual
  navigation capsules, role-specific sign-in, patient/doctor Home, profile and
  timeline, image upload and integrated map/report controls.
- `service/test_dashboard.py`, `test_security.py`, `test_parity.py`,
  `browser_check.py`: adapt existing checks to real accounts and new navigation.
- `.github/workflows/checks.yml`, `.env.example`, `README.md`, `WEB_APP.md`,
  `DEPLOYMENT.md`, `MODEL_CARD.md`: checks, configuration and current instructions.

No new runtime framework or dependency was required. PDF generation uses the
already-installed Matplotlib PDF backend. The existing model-loading,
prediction, quality, colour and Grad-CAM algorithms and trained weights were
not modified or retrained for this redesign.

## Remaining limitations

Not a validated clinical device; no new clinical-accuracy claim is made. Email
verification/recovery, MFA, external identity-provider integration, scheduled
reminders, automated retention/deletion, a tamper-proof clinical audit system,
and cloud deployment are not implemented. The account database audit covers
selected mutations, not every record read. The audit remains editable by a
host administrator. PDF text uses the installed fonts; a dedicated multilingual
font/localization pass is still needed. Docker/TLS deployment has not been tested
on this workstation. Browser tests cover Chrome desktop/mobile layouts, not
physical Android/iPhone devices or every assistive technology.
