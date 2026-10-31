# WoundWatch application

WoundWatch is now a patient-first monitoring application with real patient and
clinician accounts. The existing FastAPI, SQLite, PyTorch and plain-JavaScript
architecture is retained. No model algorithms or weights were changed by the
interface redesign.

**Read [PATIENT_GUIDE.md](PATIENT_GUIDE.md)** for exact start commands, accounts,
linking old patient records, heatmap/Grad-CAM steps, file inventory and testing.
Read [DEPLOYMENT.md](DEPLOYMENT.md) for host deployment requirements and limits.

## Run locally

```powershell
.\start.ps1
```

Open http://127.0.0.1:8000. Stop an old server with Ctrl+C first. If you see a
backend-version warning or missing visualization routes, restart the Python
server and refresh the browser with Ctrl+Shift+R.

Patient registration creates a new private patient record. Clinician accounts
and links to existing patient records require the administrator commands in
PATIENT_GUIDE.md. There is no default password and no fake role selector.

## Navigation

Both patients and healthcare workers use the same compact **Home / My Profile /
Settings** navigation capsule. Notifications are available from the header
bell, rather than a separate page. Patients can start a scan from the Home
screen's prominent **Scan My Wound** action or the persistent mobile action.
Their AI analysis, risk status, scan history, colour comparisons and reports
are available within their wound journey.

Healthcare workers see their authorised patient list and recent scans on Home.
They can search the list, open an assigned patient's profile and review that
patient's wound journey. **My Profile** for a healthcare worker shows the
worker's own account details, not a patient record. Only assigned patients are
listed.

History, colour comparisons and reports are integrated into **My Profile (or
an authorised patient profile) → My Wound Journey → View details**. Each wound
has a separate timeline. The app uses the model's actual Low/Elevated/Defer/
Retake outputs and does not invent a Moderate/High three-class classifier.

## Existing models

| Component | Implementation |
| --- | --- |
| Active classifier | ResNet-18 with two-class head; `models/classifier.pt` |
| Classifier target | SurgWound Low vs combined Medium+High (Elevated) research labels |
| Segmenter | DeepLabV3+ / ResNet-18; `models/segmenter.pt` by default |
| Input | Stored RGB orientation; no EXIF rotation |
| Preprocessing | Existing bicubic resizing, 224×224 classifier / 256×256 segmenter; ImageNet normalization |
| Quality gate | Blur, clipped pixel fraction, predicted wound area |
| Result | Retake/defer/ok, optional calibrated Elevated probability, risk band, reasons |
| Colour | Existing CIELAB wound/ring measurements, versioned bounded-resolution extraction |
| Drift | Differences between matched features of two scans of the same wound |
| Overlays | Fixed-scale raw a* heatmap; Elevated-logit Grad-CAM |
| Lighting correction | Not implemented |

The browser sends multipart photographs to `/api/scans`. The service reuses
`predict_image` and the existing colour pipeline, stores actual outputs in
SQLite, and saves a metadata-free PNG under the private runtime directory.
The original `/predict` endpoint remains available to signed-in accounts with
a CSRF header; it does not store uploads. A retake skips classification.

Colour extraction can still return raw measurements on a retake image with a
warning, but never substitutes whole-image statistics for an empty wound/ring.
Heatmaps are not infection maps. Grad-CAM cannot explain a skipped classifier,
an unverified historical classifier version or an all-zero positive activation.
Older scans can be re-analysed into a new record while preserving originals.

## Accounts and access

Patient accounts own one patient record. Clinician accounts access only explicit
assignments. Checks protect lists, profile updates, wound sites, uploads,
history, images, comparisons, reanalysis, reviews, notifications and PDFs.
Patients cannot mark a scan as professionally reviewed or create clinician
accounts. Assignment changes take effect on subsequent API requests.

Passwords use salted PBKDF2-HMAC-SHA256 with 600,000 iterations. Sessions are
random tokens, stored as hashes in SQLite and sent in HttpOnly/SameSite=Strict
cookies (Secure under production HTTPS). Writes require a per-session CSRF
token. Sessions expire after eight hours; password changes revoke all sessions.
Login/register limits and existing host/origin/body guards are enforced.
No passwords, session tokens or patient records are stored in localStorage.

References: [OWASP password storage guidance](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
and [session-management guidance](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html).
These are implemented controls, not a claim of certification or exhaustive
security review. Self-registration does not verify ownership of an email address;
administrators must verify identities before linking existing records.

## Data preservation

Existing `patients`, `scans` and `events` tables and image files remain in place.
Additive tables hold users, assignments, sessions, wounds, per-user read markers,
rate-limit buckets and selected change audits. Legacy patient records start
unassigned; they are never automatically granted to the first new account.
Existing scan JSON is interpreted with a stable wound ID derived from patient,
surgery date and wound location, without a destructive migration.

Default storage: `runtime/woundwatch.sqlite` and `runtime/images/`. Both are
excluded from Git. Images and reports are served only through authorised API
routes. New scans append history. Nothing is automatically deleted; the
organisation must define and administer its retention policy. Back up SQLite
and images together consistently, with encryption appropriate to the data.

## Reports and verification

The protected `/api/scans/{id}/report.pdf` route generates a real downloadable
PDF with the installed Matplotlib PDF backend. It includes the selected scan's
photograph, snapshot details, actual output, colour comparison and same-wound
history. Browser printing is also available. No new runtime dependency is used.

Automated checks cover existing inference, colour references, real model/API
parity, patient isolation, clinician assignments, cross-wound rejection,
session/CSRF behaviour, legacy reanalysis, heatmaps and PDF generation. Chrome
checks exercise patient and clinician views, upload two real images, generate
both overlays, download a PDF and check mobile layout/navigation. See the guide
for exact commands. The integration tests use isolated databases, not your
private records. GitHub Actions includes these checks for future pushes.

## Incomplete capabilities

No clinical diagnostic validation, new accuracy claims, email verification,
email password recovery, MFA, identity-provider integration, scheduled reminders,
automated data deletion or tamper-proof clinical audit trail. Cloud/TLS deployment
and physical Android/iPhone testing remain unverified. The SQLite audit is a
basic record of selected changes, not a complete access audit.

The earlier retraining experiment remains documented in DEPLOYMENT.md and its
aggregate report; it is separate from this UI change. Default model weights are
preserved. Candidate selection, if explicitly wanted for research, is still
available through the existing environment setting/launcher switch.
