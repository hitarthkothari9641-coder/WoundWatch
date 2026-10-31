# WoundWatch

Python/PyTorch research pipeline for wound-image experiments. It is an
experimental research aid, not a diagnosis or a substitute for clinical review.
The scripts run from the repository root and derive data/model paths from
`python/woundwatch/config.py`.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r service\requirements.txt
$env:PYTHONPATH = "$PWD\python"
```

For an NVIDIA GPU, install the PyTorch build matching the CUDA version on your
machine by following the official PyTorch selector, then install the remaining
requirements. The scripts automatically choose CUDA when available and CPU
otherwise. If torchvision cannot download pretrained ResNet-18 weights, the
training scripts warn and use random initialization; the checkpoint records
whether ImageNet weights were used.

## Run order

```powershell
python -m woundwatch.audit_data
python -m woundwatch.train_classifier --epochs 8
python -m woundwatch.train_segmentation --epochs 25
python -m unittest discover -s python\tests -v
```

The classifier uses stratified image-level 5-fold CV (there are no patient IDs
for grouped splitting), geometric augmentation only, weighted cross-entropy,
OOF AUROC with bootstrap CI, AUPRC and prevalence baseline, calibration,
Brier score, and Youden/sensitivity thresholds. Training produces
`models/classifier.pt`, `results/cv_results.csv`, and a calibration plot.

For a separate ResNet-18 model-selection benchmark that keeps the supplied
`test` split out of training, run:

```powershell
python -m woundwatch.benchmark_classifier
```

This performs nested CV on the supplied `train`+`val` partitions, selects a
regularized logistic head over frozen ImageNet features, and evaluates the
supplied `test` split once. It saves `models/classifier_candidate.pt` and
`results/classifier_candidate_benchmark.json`. The checkpoint is a research
candidate and does not replace `models/classifier.pt` used by the service. The
test split is image-level and has no patient IDs; do not describe it as
patient-independent or use it repeatedly to tune models.

An auxiliary wound-type transfer experiment is also available:

```powershell
python -m woundwatch.kaggle_transfer --pretrain-epochs 6 --finetune-epochs 5 --folds 5
```

It removes the Kaggle archive's `mirrored_*` copies, pretrains a 10-class
wound-type model, then compares ImageNet versus Kaggle initialization on the
same SurgWound train+val folds. Kaggle wound types are not infection labels.
It creates OOF analysis, geometric augmentation previews, and before/after
Grad-CAM overlays. It does not access the supplied test split. Review
`PRODUCTION_READINESS.md` before considering any deployment.

Segmentation is a DeepLabV3+ decoder with a ResNet-18 encoder, trained on
FUSeg masks using cross-entropy plus Dice loss. Masks with any positive value
(including 1 and 255) are treated as wound. It reports FUSeg validation and
Medetec external-test Dice/IoU and saves overlays to `results/seg_examples`.

After both training scripts complete:

```powershell
python -m woundwatch.predict path\to\image.jpg
python -m woundwatch.predict path\to\image.jpg --quality-only
python -m woundwatch.export_onnx
python -m woundwatch.verify_onnx
```

`predict` returns `ok`, `retake`, or `defer`, calibrated elevated probability,
risk band, and reasons. The API is `service/app.py` (`GET /health`,
`POST /predict`, multipart field `file`). Run it locally with:

```powershell
uvicorn service.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

To build the CPU-only container and mount local checkpoints:

```powershell
docker build -f service\Dockerfile -t woundwatch-service .
docker run --rm -p 8000:8000 -v "${PWD}\models:/app/models:ro" woundwatch-service
```

The exported ONNX graphs accept NCHW `float32` input already resized by
Pillow bicubic, converted to `[0,1]`, and ImageNet mean/std normalized using
`(0.485,0.456,0.406)` / `(0.229,0.224,0.225)`. The classifier ONNX output is
`[Low, Elevated]` probabilities; the segmenter output is two-class logits.

Optional service parity check: create a CSV from local Python inference, start
the API, then compare it with the HTTP outputs:

```powershell
python -m woundwatch.save_parity_reference path\to\image1.jpg path\to\image2.jpg
$env:PYTHON_PARITY_CSV = "$PWD\service\python_parity.csv"
python -m unittest service.test_parity -v
```

Color features for a longitudinal table (`patientID,day,file`) and exploratory
erythema AUROC:

```powershell
python -m woundwatch.color_features visits.csv
python -m woundwatch.test_erythema
```

The Kaggle folder is also used for the auxiliary wound-type transfer experiment
documented in `results/KAGGLE_TRANSFER_REPORT.md`; its wound-type labels are not
infection labels. Dataset licenses and redistribution rights remain unverified.
Do not redistribute Kaggle-derived weights, panels, or images until permissions
are confirmed.
