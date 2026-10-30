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

Initial file audit covered 4,220 images: SurgWound 686, Kaggle 2,940, and
Medetec 594. It found no unreadable files, exact-duplicate rows 0/2/2
respectively, and average-hash near-match rows (Hamming distance <=6) 178 / 1,018
/ 205. Near matches need visual review; they were not automatically removed
from classifier folds.

The classifier uses ImageNet-pretrained torchvision ResNet-18 weights if
available, weighted cross-entropy, geometric augmentation only, and a final
all-data fit after OOF evaluation. If pretrained weight retrieval fails, the
script warns and runs with random initialization. The checkpoint records
`pretrained_weights_used`. FUSeg segmentation uses cross-entropy plus soft
Dice loss; mask foreground is any positive pixel value, handling both 0/1 and
0/255 masks.

## Results

The completed initial classifier run used ImageNet-pretrained ResNet-18,
8 epochs per fold, image-stratified 5-fold CV, and a final 8-epoch fit on all
686 images. Pooled OOF results:

- AUROC 0.7792 (stratified bootstrap 95% CI 0.7377–0.8213).
- AUPRC 0.4029; Elevated prevalence baseline 0.1647.
- Sensitivity 0.3363 at specificity 0.9058.
- Cross-fitted Platt Brier score 0.11936 (mean squared probability error).
- Calibrated Youden threshold 0.13271; sensitivity 0.8407, specificity
  0.6370; confusion matrix TN=365, FP=208, FN=18, TP=95.
- Calibrated threshold targeting sensitivity >=0.90: threshold 0.09532;
  achieved sensitivity 0.9027 and specificity 0.4869.

These are exploratory image-level CV results, not clinical performance
claims. Segmentation validation/external metrics and ONNX parity remain pending
until those stages are run. Metrics are saved to `results/cv_results.csv` and
embedded in `models/classifier.pt` after the final fit.

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
