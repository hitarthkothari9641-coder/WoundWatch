# WoundWatch local web application

**Colour-analysis update:** See [DEPLOYMENT.md](DEPLOYMENT.md) for heatmaps,
Grad-CAM, linking/recalculating older scans, retraining and deployment controls.
The older implementation notes below describe the initial dashboard; the
update changes colour processing and adds optional operator authentication.

## Start here (Windows PowerShell)

Run from the repository root. Python 3.14 was available for this implementation;
Python 3.11+ with compatible torch/torchvision wheels also works.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r service\requirements-dev.txt
.\start.ps1
```

Open **http://127.0.0.1:8000**. The frontend is plain HTML, CSS and JavaScript
served by FastAPI; there is no npm installation or frontend compilation step.
If PowerShell blocks scripts, run the equivalent commands directly:

```powershell
$env:PYTHONPATH = "$PWD\python"
.\.venv\Scripts\python -m uvicorn service.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

Stop with Ctrl+C. Keep local mode bound to localhost. Optional operator
authentication and production guards are documented in DEPLOYMENT.md;
encryption at rest, multi-user permissions and clinical audit signatures remain
deployment requirements. Use research aliases where possible.

## Try a photograph

1. Open Patients, click Add patient and enter an alias, surgery type/date,
   wound location and monitoring status. Age is optional.
2. Open Wound Scans, select the patient, confirm dates/location and browse or
   drop a JPG/PNG. A local preview appears before submission.
3. Click Analyse wound. The server decodes and validates the image, loads the
   existing checkpoints and runs the existing prediction function.
4. Read AI Analysis and Risk Assessment. A retake means the quality gate
   rejected the photo; no probability is created. A defer is the existing
   model's uncertainty band around its decision threshold.
5. Upload another photo of the same wound/surgery and choose its previous
   scan. Colour Drift Analysis displays real measurements and their difference.
6. Use Wound History to compare two saved scans. Reports provides browser
   printing / Save as PDF and a JSON download of the stored record.

For a technical smoke test, you can use an existing image under
`data/raw/surgwound/images/train/`. This verifies execution only: those images
were used in model development and cannot demonstrate clinical performance.
No dataset images are seeded as patient records.

## What the repository actually contains

The initial inspection covered root documentation, service, all Python modules,
model artifacts, result summaries and the nested dataset inventory. The supplied
WoundAI HTML design reference was not attached; the UI follows the requested
palette and dashboard structure instead.

| Question | Finding |
| --- | --- |
| Active classifier | torchvision ResNet-18, two-class head, `models/classifier.pt` |
| Purpose | Estimate SurgWound Low vs combined Medium+High (Elevated) research labels |
| Segmenter | Existing DeepLabV3+ with ResNet-18 encoder, `models/segmenter.pt` |
| Input | RGB photographs, retaining stored orientation, without EXIF rotation |
| Preprocessing | Existing Pillow bicubic resizing, 224×224 classifier / 256×256 segmenter, ImageNet normalization |
| Quality gate | Blur variance, clipped pixel fraction, predicted wound-area fraction |
| Prediction | Existing `predict_image`: retake / defer / ok, optional calibrated Elevated probability, risk band, reasons |
| Colour | Existing `feature_row`: CIELAB wound/ring means and medians, wound-minus-ring a* |
| Drift | Subtract corresponding features between linked scans; not classifier input |
| Lighting correction | Not implemented |
| Backend | Existing FastAPI `/health` and `/predict`, preserved |
| Prior persistence | None for clinical records; added SQLite plus disk images |
| Weights | Both active .pt checkpoints and optional ONNX artifacts present; unchanged |
| Candidate | `classifier_candidate.pt` remains separate and is not promoted |

Model loading, architecture, preprocessing, calibration and thresholding remain
in the original Python package. The web service calls those functions directly.
There is no simulated output, new risk threshold or replacement external model.
Retraining now creates a separate segmenter candidate without changing active
weights. CIELAB extraction is independent of the classifier quality gate; raw
measurements on retake photos carry a warning. An empty wound/ring is unavailable.
Segmentation runs again at bounded resolution for colour feature extraction.

The classifier is not a validated post-surgical infection diagnosis model.
It does not distinguish Moderate from High. The overview therefore shows
**Elevated-risk cases**, defined as patients with an elevated latest scan for at
least one surgery/wound, instead of inventing a High category. No confidence
score is manufactured. External segmentation performance is poor in the
existing model card; colour summaries inherit segmentation and lighting errors.

## Architecture and files

The browser calls same-origin `/api/...` routes with fetch. Multipart uploads
go to `/api/scans`; the backend calls `predict_image` and `feature_row`, stores
structured JSON in SQLite, and stores a metadata-free PNG separately. No
patient information is stored in browser localStorage or sent to third-party
services. The app uses local SVG icons, system fonts and CSS/SVG charts.

New files:

- `service/dashboard.py`: patient CRUD, scan workflow, comparisons, review,
  notifications, image serving and settings.
- `service/store.py`: SQLite persistence and disk image directory.
- `service/web/index.html`, `style.css`, `app.js`: responsive ten-page dashboard.
- `service/test_dashboard.py`: isolated tests with real checkpoint inference.
- `service/browser_check.py`: optional Chrome end-to-end verification.
- `service/requirements-dev.txt`: test client dependency.
- `start.ps1`, `.env.example`, this guide.

Modified: `service/app.py` (shared validation, router/static mounting, model
loading lock, threaded prediction, response headers), `service/Dockerfile`
(include dashboard files), `.gitignore` (exclude runtime records), `README.md`
(application entry point), `MODEL_CARD.md` (document opt-in scan persistence).
Active model weights remain unchanged. The segmentation dataset now accepts an
optional resize interpolation override used by the separate retraining script.

## Persistence and settings

Records are saved under `runtime/woundwatch.sqlite`; photographs under
`runtime/images/`. Both are excluded from Git. Back up the database and images
together with the server stopped. Dates, patient linkage and prior-scan
ownership are validated server-side. Comparisons require the same patient,
surgery date, wound location and an earlier/equal scan date. Patient details are
snapshotted at scan time for reports. A review flag records a local action,
not an authenticated clinician attestation.

Optional environment variables: `WOUNDWATCH_DATA_DIR`, `MAX_UPLOAD_BYTES`
(default 10 MiB), `MAX_IMAGE_PIXELS` (default 20 million). `.env` is not loaded
automatically by start.ps1; use `--env-file .env` with uvicorn if desired.
Inference runs serially to bound simultaneous model memory use.
The original `/predict` endpoint still processes without storing uploads.
Only `/api/scans` creates persistent records.

## Verification

```powershell
$env:PYTHONPATH = "$PWD\python"
.\.venv\Scripts\python -m unittest discover -s python\tests -v
.\.venv\Scripts\python -m unittest service.test_dashboard -v
.\.venv\Scripts\python -m unittest service.test_security -v
node --check service/web/app.js
```

Tests use an isolated temporary database, compare `/predict` and stored scans
against actual direct checkpoint inference, test retakes, upload rejection,
patient validation/editing, history persistence, review and notifications.
They never populate your real workspace with fake records.

Verification completed during implementation: all four existing pipeline tests
and all five dashboard integration tests passed. An actual repository photograph
produced probability `0.4200289114489168`, identically through direct inference,
`/predict`, and the stored-scan workflow. Real CIELAB features were available;
uploading the same photograph twice produced zero feature drift. Retake and
error paths were tested separately.

Chrome desktop (1440 px) and mobile (390 px) checks passed: patient creation,
two actual model uploads, colour comparison, history, navigation, JSON download
and PDF rendering, with no browser console or page errors and no mobile page
overflow. Screenshots and the rendered PDF are in ignored `test-results/`.
Browser-test records are isolated under `test-results/browser-runtime`, not
the default `runtime/` workspace. The verification server was stopped afterward.

To repeat browser checks, install `playwright` in the virtual environment, start
the server with `WOUNDWATCH_DATA_DIR` pointing to a separate test directory,
then run `.\.venv\Scripts\python service/browser_check.py`. Installed Chrome is
required. This script creates test records and should not target real records.

## Supported and remaining work

Implemented: responsive navigation, search/filter/add/edit patients, real
uploads and inference, stored stage outcomes, scan history, side-by-side
comparison, CIELAB trends, research screening views, notifications from actual
scan events, local review tracking, print/PDF and JSON reports, settings and
model health check. All dashboard numbers derive from saved records.

Not implemented: clinical diagnosis, validated infection-free claims,
lighting correction, patient-disjoint prospective validation, scheduled
follow-up reminders, multi-user authentication (operator authentication is available), encryption/key management,
authenticated audit trails, automatic retention/deletion, cloud deployment,
server-side PDF rendering or background report jobs. The UI labels unsupported
capabilities. Production deployment requires those privacy/security controls
and dataset/model licensing review, plus clinical validation appropriate to use.
