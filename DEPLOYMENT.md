# Deployment preparation and release limits

This release fixes the colour workflow and adds real colour heatmaps and
Grad-CAM. It is a hardened **research application**, not a clinically validated
production medical device. Do not represent software deployment controls as
evidence that infection screening is clinically accurate.

## Local use and account migration

Read [PATIENT_GUIDE.md](PATIENT_GUIDE.md) for the current sign-in and account
setup. The patient-first redesign adds mandatory patient/clinician sessions and
backend record permissions. Existing records remain intact but unassigned until
an administrator explicitly links them; no destructive migration is performed.

Restart the server with `start.ps1`, sign in, and open **My Profile ? My Wound
Journey ? View details ? Colour comparison**. Generate overlays under **Explore
colour heatmap and AI attention map**. For older records, **Re-analyse saved
image** creates a new versioned result without overwriting the original.

Choose an earlier photo of the same wound to compare. Recalculate colour on
both records if their measurement or segmenter versions differ. Reports are
available in the selected scan's **Reports** tab as downloadable PDFs.

Colour extraction is now independent of the classifier quality gate. Retake
photos can display raw measurements if a wound/ring is measurable, with a
quality warning; they still receive no classifier probability or Grad-CAM.
An empty predicted wound/ring remains unavailable, rather than silently
substituting the whole image as a wound. The full-image colour heatmap works
even when wound segmentation cannot yield summary features.

The colour pipeline uses an aspect-preserving 512-pixel maximum dimension and
a ring width scaled from 12 pixels at 256 pixels. This bounds memory and reduces
resolution-induced measurement changes. Comparisons reject mixed measurement
methods or segmenter versions. Recalculation preserves the original screening
result and refreshes dependent comparisons. It does not improve that result.

Colour overlays map measured CIELAB a* to a fixed range of −20 to +60, clipped
at its ends. They show red–green appearance, not infection or between-image
pixel drift. Photographs are not registered; reporting pixelwise longitudinal
differences would be misleading with different camera positions.

Grad-CAM uses gradients of the raw Elevated logit at ResNet-18 layer4, spatial
gradient averaging, weighted activations, ReLU and per-image normalization.
It does not alter weights or parameter gradients. Older scans without a
recorded classifier fingerprint cannot receive a traceable historical Grad-CAM:
use Re-analyse saved image to create a new versioned analysis. Flat positive activation is reported as unavailable. This coarse
explanation can be misleading and is not a diagnostic mask. Method reference:
[original Grad-CAM paper](https://arxiv.org/abs/1610.02391).

## Controlled research deployment

Named patient/clinician accounts and server-side record authorization are now
mandatory. Sessions use hashed random tokens, HttpOnly/SameSite cookies, expiry
and CSRF validation. Patient registration cannot create clinician privileges;
clinicians and assignments are provisioned through `service.manage`. See the
patient guide for all commands. Optional HTTP Basic credentials are an extra
service gate only; they do not replace a named account or grant record access.

Retained controls include production host/HTTPS guards, request-size limits,
no-store responses, model fingerprints, SQLite WAL and a non-root container.
Production uses Secure cookies and HSTS. Selected mutations are audited with
account IDs, but the audit is not tamper-proof and does not cover every read.
MFA, email verification/recovery and external identity integration remain
unimplemented. These controls do not establish clinical production readiness.

1. Install Docker and Compose on a reviewed host. Docker was unavailable in
   the implementation environment, so container build/TLS deployment remains
   unverified; the manifests were syntax-checked only.
2. Copy `.env.example` to `.env`. Set `WOUNDWATCH_ENV=production`,
   `WOUNDWATCH_DOMAIN` to your DNS name, and `WOUNDWATCH_ALLOWED_HOSTS` to that
   hostname plus `127.0.0.1`. Provision clinician accounts against the mounted
   data volume with `python -m service.manage`. Never commit `.env` or passwords.
3. Set `MAX_IMAGE_PIXELS=4000000` for a conservative memory bound and establish
   memory/concurrency limits appropriate to your workload through load tests.
4. Confirm dataset/weight redistribution and deployment permissions before
   making the service available. Mount approved checkpoints under `models/`.
5. Run `docker compose up --build -d`. Caddy terminates HTTPS. Only the proxy
   exposes ports; Uvicorn trusts forwarded headers only within this isolated
   network. Do not expose the application port directly or reuse unrestricted
   forwarded-header trust on a public interface.
6. Verify HTTPS, unauthorized requests returning 401, authorized model health,
   image upload, persistence across restart and restore from backup before use.

The Docker volume persists SQLite and images together. Use encrypted host
storage and encrypted backups. Stop the service or use a consistent SQLite
backup procedure; copying only the database file while WAL writes are active
is not a reliable backup. Define retention and deletion policies before storing
identifiable information. Do not log request bodies, authorization headers or
patient identifiers at the reverse proxy.

HTTPS architecture follows [FastAPI deployment guidance](https://fastapi.tiangolo.com/deployment/https/).
This repository does not configure a public domain, a hosted identity provider,
disk encryption, recovery/retention policy or clinical validation automatically.

## Model retraining

`python -m woundwatch.retrain_segmenter --epochs 2 --batch-size 1 --output experiments/my-run`

This retrains the existing segmenter decoder with a frozen encoder, conservative
learning rate, frozen batch-normalization statistics and gradient clipping.
It uses only the 810 FUSeg training images, selects on 200 FUSeg validation
images, and reports the selected model on the 8-image external Medetec set.
Inference-matched bicubic resizing is used in the experiment. Existing training
defaults and deployed weights are preserved. Checkpoints are saved each time
validation Dice improves, and the final candidate and JSON report are separate.

Selection against the same historical validation set is exploratory and not
an independent clinical evaluation. These are foot-ulcer datasets, not labeled
surgical-wound segmentation data. No classifier infection-accuracy improvement
can be inferred from segmentation Dice. The active classifier is unchanged.

Candidate weights remain local under ignored `experiments/`; code and aggregate
results can be reviewed on GitHub without publishing additional derived weights.
To test an approved candidate explicitly, set `WOUNDWATCH_SEGMENTER_PATH` to its
checkpoint before starting the server. Recalculate both scans' colour features
after switching segmenter versions. Revert by clearing that environment variable.

The completed two-epoch experiment produced:

| Metric | Original, re-evaluated | Retrained candidate |
| --- | ---: | ---: |
| FUSeg validation Dice (200 images) | 0.8310 | 0.8528 |
| External Medetec Dice (8 images) | 0.2358 | 0.4162 |
| External Medetec IoU | 0.1337 | 0.2628 |

See `results/segmenter_retrain_report.json` for the complete aggregate report.
These baseline values differ from the historical model card because this
experiment uses inference-matched bicubic resizing for both models. The
original and candidate are compared with identical settings in this report.
An initial batch-size-4 run ran out of RAM during epoch 3; the reported run was
restarted from the original checkpoint with batch size 1 and two epochs.
The external Dice remains too low to claim reliable surgical-wound segmentation.

On this workstation, the completed candidate can be tried using:

```powershell
.\start.ps1 -UseRetrainedSegmenter
```

This selects the local candidate explicitly; it does not overwrite the original
checkpoint. Stop the server, clear `$env:WOUNDWATCH_SEGMENTER_PATH`, and run
`start.ps1` to return to the original. Fresh GitHub checkouts must run the
retraining command first because the additional candidate weights are not
published with this code release.

## Tests and outstanding release work

Run the existing pipeline tests, new colour-reference tests, dashboard tests,
security tests, and Chrome workflow described in `WEB_APP.md`. GitHub Actions
now runs Python and JavaScript checks on pushes/PRs.

Local verification: six pipeline/colour tests and eight dashboard/security
tests passed. The six dashboard integration tests also passed with the
retrained candidate selected. Chrome generated both overlays and exercised
comparison relinking/recalculation, two real model uploads, patient creation,
history, reports and mobile navigation with no page/console errors. CPU thread
defaults were reduced to two and the launcher/container limit OpenBLAS threads
to one after memory pressure appeared during concurrent test processes.

Before clinical production: acquire representative consented longitudinal
surgical-wound photos with masks and outcomes; evaluate patient-disjoint cohorts
and skin-tone/device subgroups; clinically validate operating thresholds; obtain
applicable review/approvals; complete organization-managed identity/MFA, comprehensive audit and
data lifecycle controls; complete load, penetration, backup/restore and deployed
TLS tests. These are still release blockers, not completed features.
