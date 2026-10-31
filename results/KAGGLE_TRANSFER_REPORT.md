# Kaggle auxiliary-pretraining experiment report

**Conclusion:** the Kaggle wound-type pretraining did not improve infection-label ranking in this exploratory comparison. Keep the ImageNet-initialized candidate for research; do not promote either checkpoint for clinical use.

## Protocol

- Auxiliary source: 1470 original Kaggle images from 10 wound-type folders; 1470 `mirrored_*` copies were excluded by filename pairing. These labels describe wound types, not infection.
- Target: SurgWound Low versus Medium+High; train+validation development images only, n=549, Elevated=93 (16.9%).
- Matched stratified 5-fold image-level OOF. Same folds, preprocessing, geometric-only augmentation, weighted cross-entropy and 5 fine-tuning epochs for both initialization strategies. Supplied test partition was not read by this experiment.
- CIs below are 2,000 stratified image-bootstrap replicates of pooled OOF rows. They ignore patient clustering and training variation; they are descriptive only.

## Results

| Initialization | AUROC (95% image-bootstrap CI) | AUPRC (95% image-bootstrap CI) | Calibrated Brier | Youden sensitivity | Youden specificity |
|---|---:|---:|---:|---:|---:|
| ImageNet | 0.725 (0.666?0.782) | 0.428 (0.343?0.532) | 0.123 | 0.613 | 0.719 |
| Kaggle wound-type pretraining | 0.695 (0.630?0.756) | 0.438 (0.350?0.535) | 0.127 | 0.430 | 0.890 |

Elevated prevalence baseline for AUPRC: 0.169. Kaggle minus ImageNet AUROC: -0.030; AUPRC: +0.010. The AUROC and Brier score were worse after Kaggle pretraining; a small AUPRC increase does not establish a meaningful benefit. The Youden operating point is selected on these same OOF labels and is not a deployment threshold.

## Interpretation and limits

This is a model-selection experiment on image-level folds, not patient-independent validation: patient IDs are unavailable and related images may cross folds. SurgWound has few Elevated cases and web/challenge imagery. These scores do not show performance on unseen patients, hospitals, phone cameras, skin tones, or prospective clinical workflows. Grad-CAM panels show score sensitivity, not causal explanations. Kaggle dataset redistribution rights are unverified (`licenseName: Unknown` in the local API audit); keep derived checkpoints and image panels local until the dataset terms are confirmed.

## Artifacts

- `results/kaggle_transfer_comparison.png`
- `results/augmentation_before_after.png` (geometric transforms only)
- `results/kaggle_mirror_audit.png`
- `results/transfer_gradcam/` (five matched-fold input/Grad-CAM triptychs)
- `results/kaggle_transfer_oof.csv` and `results/kaggle_transfer_analysis.json`
- `models/classifier_transfer_candidate.pt` (ImageNet initialization selected by development OOF AUROC; research candidate)
- `models/kaggle_woundtype_resnet18.pt` (auxiliary wound-type model; rights unverified)
