# WoundWatch model card

## Intended use

WoundWatch is a Python/PyTorch research prototype for image-based experiments
estimating the supplied SurgWound binary label (`Low` versus `Medium`+`High`).
It is not a diagnosis, medical device, or substitute for examination by a
qualified clinician. Do not use it to delay care or make treatment decisions.
No prospective clinical validation has been completed.

The pipeline includes a ResNet-18 image classifier, a DeepLabV3+ decoder with
ResNet-18 encoder, an image-quality gate, calibrated probability output, and
CIELAB wound/ring summaries. The service runs the saved PyTorch checkpoints;
optional ONNX exports are provided for experimentation.

## Data and training

- SurgWound: 686 images (Low 573, Medium 90, High 23), no patient IDs. The
  binary classifier uses stratified image-level 5-fold CV, so related images
  cannot be excluded across folds.
- FUSeg: foot-ulcer images and masks for segmentation training/validation.
- Medetec FUSeg test split: external segmentation evaluation.
- Kaggle Wound and Medetec folders: data audit only; class names are not
  interpreted as infection labels.
- DFUC2021: present locally, but not automatically used. Its label semantics
  differ from SurgWound; cross-dataset infection pretraining is not implemented
  in this Python conversion.

The classifier uses ImageNet-pretrained torchvision ResNet-18 weights if
available, weighted cross-entropy, geometric augmentation only, and a final
all-data fit after OOF evaluation. If pretrained weight retrieval fails, the
script warns and runs with random initialization. The checkpoint records
`pretrained_weights_used`. FUSeg segmentation uses cross-entropy plus soft
Dice loss; mask foreground is any positive pixel value, handling both 0/1 and
0/255 masks.

## Results

Metrics are written by the training scripts to `results/cv_results.csv`,
`results/segmentation_metrics.json`, and printed in the training log. No
performance values are reported here until those artifacts are produced by a
completed run. Do not substitute training metrics or literature values for
held-out evaluation.

The classifier reports pooled OOF AUROC with bootstrap 95% CI, AUPRC and
elevated-label prevalence baseline, sensitivity at >=90% specificity,
cross-fitted Platt-calibrated Brier score, Youden threshold, and a threshold
selected for sensitivity >=0.90. Segmentation reports validation and external
Medetec Dice and IoU. ONNX parity is checked separately at maximum absolute
output difference <=1e-3.

## Preprocessing and inference

Images are decoded to stored RGB orientation without applying EXIF orientation,
resized with Pillow bicubic interpolation to 224x224 (classifier) or 256x256
(segmenter), scaled to `[0,1]`, then ImageNet mean/std normalized. Training and
service use this same preprocessing implementation. The classifier's raw
Elevated softmax score is transformed with Platt intercept/slope fit on OOF
scores. `predict` returns `retake` when quality checks fail, `defer` near the
Youden threshold, otherwise `ok` plus the risk band. These are workflow labels,
not care recommendations.

## Known limits

- Small, imbalanced dataset; only 23 High examples.
- No patient IDs; image-level CV can leak related images across folds.
- Web and challenge images may not represent smartphone photos from surgical
  care settings.
- No skin-tone labels; subgroup performance/calibration cannot be assessed.
- Lighting, camera processing, dressings, wound location, time since surgery,
  and clinical context can affect predictions.
- FUSeg and Medetec are foot-ulcer imagery, not a validated proxy for surgical
  wounds. Applying the segmenter to SurgWound is a domain shift.
- CIELAB erythema is an exploratory image feature, not a clinical biomarker;
  segmenter errors and illumination shifts affect it.
- Probability calibration may not transfer to other populations, devices, or
  acquisition workflows.

## Sources, licensing, privacy, monitoring

The local dataset copies have not all had their exact image-use rights
verified. The [SurgWound repository](https://github.com/xuxuxuxuxuxjh/SurgWound),
[FUSeg challenge](https://fusc.grand-challenge.org/FUSeg-2021/),
[DFUC2021 access page](https://dfu-2021.grand-challenge.org/Dataset/), and
ImageNet/PyTorch weight terms must be reviewed before redistribution or hosted
use. Kaggle and Medetec source license records are incomplete locally. Do not
redistribute images or derived weights until permissions are confirmed.

`service/app.py` limits upload bytes and decoded pixels, processes images in
memory, does not store uploads or log image content/filenames, and returns
generic errors. Keep reverse-proxy body logging disabled. Monitor aggregate
service health, latency, quality rejection, delayed consented outcomes, and
discrimination/calibration over time. Do not log patient identifiers or
individual predictions. Revalidate and recalibrate only on an appropriately
separated cohort before any prospective use.
