# WoundWatch model card

## Summary

WoundWatch is a research prototype that estimates the probability of an
`Elevated` post-surgical wound infection-risk label from a photograph. The
binary task maps the supplied SurgWound labels as `Low` versus `Medium` or
`High`. An ONNX segmenter estimates a wound region, a quality gate can request
a retake, and an ONNX classifier produces a score that is Platt calibrated.

**This is not a diagnosis, medical device, or substitute for examination by a
qualified clinician.** Do not use its output to delay care or make treatment
decisions. No prospective clinical validation has been completed.

## Intended use

- Research and engineering evaluation of image-based wound classification.
- Controlled offline or locally hosted testing with de-identified images and
  permissions to use them.
- The API returns `retake` for quality-gate rejection, `defer` near the
  configured threshold, and `ok` otherwise. Those statuses are technical
  workflow labels, not care recommendations.

Out of scope: diagnosis, autonomous triage, treatment selection, clinical
documentation, or use on an individual patient without appropriate clinical,
privacy, regulatory, and prospective validation review.

## Model and preprocessing

The intended classifier is the MATLAB transfer-learning model selected in
`code/config.m`; the intended segmentation model is DeepLab v3+ with ResNet-18.
The serving container loads `classifier.onnx` and `segmenter.onnx` from a
read-only mounted `/models` directory. It expects a single RGB image input per
model. NCHW and NHWC graph layouts are inferred from the ONNX input shape; set
`ONNX_IMAGE_LAYOUT` only for dynamic-shape graphs whose channel axis cannot be
inferred.

Preprocessing contract, in order:

1. Decode the uploaded file in memory. Use its stored pixel orientation; do not
   apply EXIF orientation. Convert the decoded pixels to three-channel RGB.
   Conversion creates a metadata-free image; the upload is not saved.
2. Resize to the static height and width declared by each ONNX input (fallback
   dimensions are 224x224 for classification and 256x256 for segmentation).
   Use bicubic interpolation. MATLAB training/inference uses
   `imresize(uint8RGB,[H W])`; Python uses Pillow `Image.Resampling.BICUBIC`.
   Both produce 8-bit RGB pixels. Pixel rounding/edge implementation can differ
   between MATLAB and Pillow, so the parity test is required for each model
   export. If it exceeds tolerance, do not deploy that combination.
3. Cast the resized RGB array to `float32` with values in `[0,255]`. Do not
   divide by 255, reorder RGB to BGR, or apply external mean/std normalization.
   Any normalization embedded in the exported ONNX graph is applied by that
   graph.
4. Interpret classifier outputs in `[Low, Elevated]` order as probabilities;
   compute `rawElevated`, clamp to `[1e-7, 1-1e-7]`, then calculate
   `sigmoid(intercept + slope * log(rawElevated/(1-rawElevated)))`. The
   intercept, slope, and threshold must be copied from the exact MATLAB
   `classifier.mat` artifact into service environment variables.

`train_classifier.m` normally uses pretrained weights. If pretrained loading
fails with MATLAB service error 5006, its configured fallback constructs a
local ResNet architecture using `resnetNetwork` without contacting the model
service. For ResNet-18, the local call uses stack depths `[2 2 2 2]`, filter
counts `[64 128 256 512]`, and `BottleneckType="none"`. It has random weights;
the script warns about that fallback and saves `pretrainedWeightsUsed=false`
in the MAT file. Treat such a run as training from scratch and retain that
provenance with its metrics; it is a different experiment from transfer
learning. The local fallback is implemented for `resnet18` and `resnet50`;
`efficientnetb0` still requires its pretrained architecture to load normally.

The API quality gate mirrors the MATLAB blur, exposure-clipping, and predicted
wound-area checks. Its Laplacian kernel and thresholds are documented in
`service/app.py`; verify the gate behavior on representative images before use.

## Data sources and permissions

The local workspace contains SurgWound (686 converted images, labeled Low 573,
Medium 90, High 23), FUSeg, DFUC2021, a Kaggle wound-image copy, and Medetec
images. The planned MATLAB classifier workflow uses SurgWound. The segmentation
workflow uses FUSeg, with the configured Medetec external-test subset. DFUC2021,
Kaggle, and the separate Medetec collection are not represented as training
inputs to the current classifier unless a separate training record says otherwise.

| Source | Role / provenance | License or permission status |
|---|---|---|
| ImageNet | Pretraining source for the ResNet-18 initialization loaded by MATLAB `imagePretrainedNetwork`. | [MathWorks documents](https://www.mathworks.com/help/deeplearning/ref/imagepretrainednetwork.html) the available ResNet-18 weights as ImageNet-trained. [ImageNet states](https://www.image-net.org/about.php) its images are copyrighted by their original owners and access is for non-commercial research/education subject to its terms. Check the exact MathWorks support-package weight terms before redistributing weights or using commercially. |
| SurgWound | Surgical-wound images and severity labels; local CSV labels were converted for this project. | The [SurgWound repository](https://github.com/xuxuxuxuxuxjh/SurgWound) describes dataset images as subject to separate terms from repository code. The exact image license/permission for this local copy is **not established** by the workspace. Verify source terms before redistribution or hosted use. |
| FUSeg 2021 | Foot-ulcer images and wound masks for segmentation. The local README identifies it as an open challenge and gives its [challenge source](https://fusc.grand-challenge.org/FUSeg-2021/). | The local README does not state an image license. “Open challenge” is not treated here as a grant of commercial or redistribution rights. Confirm the applicable source terms. |
| DFUC2021 | Diabetic-foot images with infection, ischaemia, both, and control labels; optional research source only. | Access is governed by the [DFUC dataset permission process](https://dfu-2021.grand-challenge.org/Dataset/); its data page describes research use and directs users to request permission. Confirm that the local copy is covered by the relevant agreement. |
| Kaggle Wound dataset copy | Local folder of wound classes, including Surgical Wounds; known duplicate/mirrored images. | The exact Kaggle dataset page and license are not recorded locally. **Unverified; do not redistribute or train for external use until identified and checked.** |
| Medetec | Local wound-image collection; used as external segmentation test according to project configuration. | The local workspace has no license/permission record. **Unverified; check the original source terms and image rights before use or redistribution.** |

The FUSeg source describes a collection of de-identified clinical images; this
does not remove license, consent, or permitted-use obligations. Data-source
licenses do not automatically grant rights to publish derived model weights.
The exact data manifests, preprocessing, and source versions should be
recorded with any future model release.

## Evaluation and metrics

No trained ONNX or MATLAB model artifacts and no `results/cv_results.csv` were
present in the workspace when this card was written. **There are no measured
performance values or confidence intervals to report yet.** Do not fill these
with training-set or literature values. Populate them from the exact model
artifact and evaluation data before considering any external evaluation.

| Evaluation | Metric | Result |
|---|---|---|
| SurgWound stratified 5-fold image-level CV | Pooled out-of-fold AUROC (bootstrap 95% CI) | Not measured; pending model run |
| SurgWound stratified 5-fold image-level CV | AUPRC and Elevated prevalence baseline | Not measured; pending model run |
| SurgWound stratified 5-fold image-level CV | Sensitivity at 90% specificity | Not measured; pending model run |
| SurgWound stratified 5-fold image-level CV | Platt-calibrated Brier score and calibration plot | Not measured; pending model run |
| FUSeg validation | Mean Dice and mean IoU | Not measured; pending model run |
| Medetec external segmentation test | Mean Dice and mean IoU | Not measured; pending model run |
| MATLAB-to-ONNX deployment parity | Maximum absolute probability difference | Not measured; run `service/test_parity.py` with a populated MATLAB reference CSV |

CV is image-level. No patient identifiers are available, so patient-level
partitioning and leakage checks cannot be performed. The reported 686 source
images are small for deep learning, especially the 23 High-label images.

## Known limits and risks

- Small, imbalanced dataset; model behavior for the High class is especially
  uncertain.
- No patient IDs in SurgWound; related images may cross folds.
- Data may include web-sourced or otherwise non-standardized photographs,
  which may not represent smartphone images captured during clinical care.
- Skin-tone labels are absent. Performance and calibration across skin tones
  cannot be assessed from the current records.
- Lighting, camera processing, wound location, dressings, background, time
  since surgery, and clinical context can change predictions.
- The FUSeg/Medetec segmentation domain is foot-ulcer imagery, not a validated
  proxy for post-surgical wounds.
- The segmentation mask and quality gate can be wrong; an `ok` status does not
  establish that an image is clinically adequate.
- Calibration can drift with population, device, acquisition, preprocessing,
  or model changes. A displayed probability is not an individualized clinical
  risk estimate until external validation supports that interpretation.
- Pillow and MATLAB bicubic resizing can differ in pixel rounding and edge
  handling; numerical parity is a release gate, not an assumption.

## Service privacy and monitoring plan

`service/app.py` enforces a byte limit and decoded-pixel limit, processes image
bytes in memory, does not write uploads, does not include image bytes or
filenames in application logs/responses, and returns generic errors. The
container disables Uvicorn access logs. Keep reverse-proxy request-body and
debug logging disabled as well; configure short-lived memory and secure host
access. Do not send patient images to a public service without an approved
privacy and security review.

Before any prospective use, establish a versioned monitoring process that:

- Logs only model version, timestamp bucket, service health/latency, status
  counts, and aggregate input-quality statistics; never log image bytes,
  filenames, patient IDs, or raw predictions tied to an individual.
- Tracks delayed, consented clinical outcomes with a defined evaluation
  protocol and compares discrimination, calibration, sensitivity, and
  specificity to a pre-specified baseline.
- Audits image quality and performance by device/site and, when ethically and
  lawfully collected with consent, skin-tone strata; do not infer skin tone
  from the image to fill the current data gap.
- Defines drift and performance alert thresholds, clinical review, rollback,
  and revalidation steps before enabling updates. Refit calibration only on
  an appropriately separated calibration cohort.
- Repeats MATLAB-to-ONNX parity tests after any model, library, preprocessing,
  hardware-provider, or container change.

## Deployment configuration

Mount the two exported model files at `/models/classifier.onnx` and
`/models/segmenter.onnx`. Set `PLATT_INTERCEPT`, `PLATT_SLOPE`, and
`DECISION_THRESHOLD` from the exact saved MATLAB classifier; the service fails
health checks when those values or models are missing. Optional settings are
`DEFER_HALF_WIDTH`, `BLUR_VARIANCE_MINIMUM`, `MAX_CLIPPED_FRACTION`,
`MIN_WOUND_AREA_FRACTION`, `MAX_WOUND_AREA_FRACTION`, `MAX_UPLOAD_BYTES`,
`MAX_IMAGE_PIXELS`, and `ONNX_THREADS`.

The CSV format expected by the parity test is documented in
`service/test_parity.py`; `service/matlab_parity_template.csv` is a header-only
template. Populate it with MATLAB `predict_ssi` probabilities and status for
the exact same images used against the service. Use only images cleared for
this purpose.
