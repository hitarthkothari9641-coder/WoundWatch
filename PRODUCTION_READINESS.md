# WoundWatch production-readiness review

**Status: research prototype. Not ready for patient care, clinical deployment, or
medical-device performance claims.** A higher image-level score cannot replace
patient-disjoint and prospective clinical evidence.

## Kaggle data and current experiment

The local Kaggle folder contains 2,940 files across 10 wound-type folders. It
has 1,470 `mirrored_*` files with a matching non-mirrored filename in the same
class folder. The training script excludes those copies and uses 1,470 source
images for an auxiliary 10-class wound-type task. It then fine-tunes the
ImageNet and Kaggle initializations on identical SurgWound train+validation
folds. Kaggle folder labels are wound categories (for example pressure,
venous, diabetic, surgical, and normal); they are **not infection labels**.
The experiment must not translate a wound category into Low/Elevated infection
risk.

Run the experiment from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\python"
python -m woundwatch.kaggle_transfer --pretrain-epochs 6 --finetune-epochs 5 --folds 5
```

It writes:

- `models/kaggle_woundtype_resnet18.pt`: auxiliary wound-type checkpoint.
- `models/classifier_transfer_candidate.pt`: candidate selected from
  train+validation OOF AUROC; it does not replace `models/classifier.pt`.
- `results/kaggle_transfer_analysis.json` and
  `results/kaggle_transfer_oof.csv`: class counts, paired comparison, and
  development OOF predictions.
- `results/kaggle_transfer_comparison.png`: ImageNet versus Kaggle
  initialization AUROC/AUPRC comparison.
- `results/kaggle_mirror_audit.png` and
  `results/augmentation_before_after.png`: source/mirror and geometric
  augmentation previews.
- `results/transfer_gradcam/`: same-fold before/after Grad-CAM overlays on
  SurgWound development examples. Grad-CAM highlights sensitivity of a model
  score to image regions; it is not a causal explanation or a clinical reason.

The script intentionally does not use the SurgWound test partition. It was
already evaluated during an earlier candidate experiment, so it is no longer a
fresh locked test set for this project. Its reported AUROC cannot validate a
new Kaggle-pretrained candidate. Results from the new experiment are
development-only and image-level; do not choose an operating threshold or
claim generalization to unseen patients from them.

## Blocking issues before any clinical use

1. **Patient linkage and outcomes:** SurgWound has no patient IDs, visit dates,
   or documented patient-level adjudication. The current labels are image-level
   Low/Medium/High labels. Repeated images cannot be assigned to groups
   reliably, and the current near-duplicate hash links include label conflicts.
   Obtain lawful patient/encounter/visit linkage and independently adjudicated
   SSI outcomes, or collect a new dataset.
2. **External surgical-wound evaluation:** the current classifier data do not
   demonstrate performance on a new hospital, population, device, or patient.
   The supplied image-level test split is small (20 Elevated images) and has
   already been viewed. Collect a separately governed, patient-disjoint and
   preferably site-disjoint test cohort. Pre-specify the primary metric,
   confidence interval, and operating point before opening that cohort.
3. **Wound segmentation:** the current FUSeg/Medetec segmenter scored Dice
   0.823 on FUSeg validation (200 images), but only Dice 0.204 on the external
   Medetec subset (8 images). It is trained on foot-ulcer data and is not
   validated for surgical wounds. Do not use its masks or derived CIELAB ring
   features as clinically reliable measurements until it passes a diverse,
   labeled surgical-wound external evaluation.
4. **Kaggle rights and provenance:** the Kaggle dataset page is
   [Collected and Categorized Wound Images Dataset](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification).
   Kaggle's dataset API currently reports its `licenseName` as `Unknown`.
   Local files do not include a verified dataset license, source-level consent,
   provenance, or patient identity policy. Confirm the dataset's current terms
   and each image source's rights before using trained weights commercially,
   hosting them, or redistributing them. Kaggle's notebook license is not a
   substitute for the dataset's license.
5. **Fairness and capture conditions:** there are no reliable skin-tone labels,
   camera metadata, color-chart references, or prospective smartphone capture
   protocol. Subgroup calibration and color measurement accuracy are unknown.
6. **Operating thresholds and workflow:** thresholds from this prototype have
   not been selected with clinicians or tied to a defined care pathway. In the
   earlier candidate evaluation, the threshold selected for 90% development
   sensitivity classified every supplied test image as Elevated. A user
   interface must never present these scores as a diagnosis or allow a low
   score to delay clinical review.
7. **Quality, safety, and security controls:** before serving real users, add
   identity and access management, rate limits, TLS, deployment secrets,
   resource/time limits, tested model/version rollback, dependency and image
   scanning, security incident response, and a privacy-preserving audit plan.
   Validate that upload bodies, proxy logs, application logs, crash reports,
   and metrics cannot retain image bytes, filenames, or patient identifiers.
8. **Clinical and regulatory path:** CDSCO published final guidance on Medical
   Device Software under the Medical Devices Rules, 2017 on 21 July 2026.
   Intended medical use can bring software within that framework; obtain
   regulatory advice on intended use, classification, licensing, quality
   management, clinical investigation, technical documentation, and
   post-market obligations. See the
   [official CDSCO guidance](https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-document-on-Medical-Device-Software-under-MDR-2017.pdf)
   and [CDSCO medical-device page](https://www.cdsco.gov.in/opencms/opencms/en/Medical-Device-Diagnostics/Medical-Device-Diagnostics/).

## Required release evidence

- Documented source, provenance, licensing, lawful basis, annotation process,
  inclusion/exclusion criteria, and patient-level split construction.
- A prespecified patient-disjoint development protocol and a separately held
  out external cohort with enough positive cases for useful confidence bounds.
- Independent evaluation of discrimination, calibration, sensitivity at a
  clinician-defined specificity, subgroup performance, image-quality failures,
  segmentation failures, and decision impact.
- Prospective clinical workflow study and clinician review before any
  deployment or care-related claim.
- A maintained quality system, risk-management file, cybersecurity process,
  controlled model changes, versioned datasets/artifacts, rollback, and
  post-market monitoring plan.

For transparent medical-imaging reporting, state the data source, study design,
and whether partitions are patient- or image-level, following the
[CLAIM 2024 update](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/).
