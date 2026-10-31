# Python classifier improvement TODO

This is a proposed experiment plan, not a claim of improved performance. No
training code, checkpoints, service defaults, or existing files are changed.

## Current baseline and why accuracy alone is misleading

The separate frozen-ResNet-18 candidate's supplied test split contains 137
images: 117 Low and 20 Elevated. At its development-selected Youden threshold
(0.2785421515), its confusion counts are TN=92, FP=25, FN=7, TP=13.

| Measure | Candidate | Always predict Low |
| --- | ---: | ---: |
| Accuracy | 76.64% | 85.40% |
| Elevated precision | 34.21% | Undefined (no positive predictions) |
| Elevated recall | 65.00% | 0.00% |
| Elevated F1 | 44.83% | 0.00% |
| Balanced accuracy | 71.82% | 50.00% |

Thus even 85% accuracy can mean missing every Elevated image. The objective
should be better Elevated detection with an explicit precision/recall tradeoff,
not a higher accuracy number alone. There are only 20 positive test images;
one changed positive prediction moves recall by 5 percentage points.

Sources: `results/classifier_candidate_benchmark.json`,
`results/classifier_candidate_test_predictions.csv`, and `MODEL_CARD.md`.
These candidate test scores are not the active model's scores. The active
`models/classifier.pt` is fitted on all 686 images after CV; evaluating that
checkpoint on the supplied test images is not a held-out evaluation.

## P0: Establish a trustworthy experiment before changing the model

- [ ] Keep the supplied test set out of training, calibration, early stopping,
  threshold selection, and hyperparameter selection. Use only train+val for
  development. Since existing test results are already visible, use a new,
  untouched cohort for confirmation after iterative development.
- [ ] Obtain patient/wound/visit identifiers if possible. Split by patient,
  retaining all visits from a patient in the same partition. Without these IDs,
  label results image-level; do not claim unseen-patient performance.
- [ ] Review exact duplicates, mirrored copies, near matches and conflicting
  labels. The existing audit reports 178 SurgWound near-match rows; these are
  review candidates, not proven duplicates. Keep confirmed related images in
  one split. Record exclusions and regenerate all baselines on the same split.
- [ ] Verify Low/Medium/High annotations with the data owner. Keep binary
  semantics fixed: Low=0, Medium+High=1. Do not relabel Kaggle wound types as
  infection labels. Prioritize additional independent Elevated examples and
  difficult Low negatives with reliable labels.
- [ ] Save split manifests, seeds, package versions, checkpoint hashes and
  dataset version for each experiment. Fail explicitly if ImageNet weights
  cannot load rather than silently comparing a randomly initialized model.

## P1: Fix selection objectives and compare thresholds

- [ ] In a separate experiment based on
  `python/woundwatch/benchmark_classifier.py`, compare model selection using
  average precision (AP) with the existing AUROC objective. Keep the frozen
  ImageNet backbone and logistic head as the inexpensive baseline.
- [ ] Predeclare the selection objective before running experiments. For a
  research comparison, use Elevated F1 plus AP and balanced accuracy. If a
  minimum recall is required, agree that requirement first and maximize
  precision subject to it; do not invent a clinical recall target.
- [ ] Select thresholds from development predictions only. Compare the saved
  Youden rule with an F1-selected threshold, or the agreed recall-constrained
  rule. In nested CV, select both calibration and threshold inside each outer
  training fold and score on the untouched outer validation fold.
- [ ] Freeze calibration and threshold before final evaluation. Increasing a
  threshold may reduce false positives but also miss more Elevated images;
  threshold tuning does not improve the underlying ranking (AUROC/AP).
- [ ] Report all four requested percentage metrics, balanced accuracy, AP,
  AUROC, prevalence, per-class support and confusion counts. Include precision-
  recall curves, calibration/Brier score and uncertainty intervals. Bootstrap
  by patient when IDs exist; otherwise explicitly label image bootstrap.

## P2: Run controlled training experiments in Colab

Proposed starting settings below are experimental, not validated improvements.
Use identical development folds, preprocessing and training budgets. Change
one factor at a time and compare at least three fixed seeds before selecting.

- [ ] Create a separate training entry point rather than modifying the active
  checkpoint. Reuse RGB decoding, bicubic 224x224 resizing and ImageNet
  normalization from `python/woundwatch/common.py`.
- [ ] Compare the frozen-feature baseline with gradual ResNet-18 fine-tuning:
  train the head first, then unfreeze layer4. Trial head learning rates of
  1e-4/3e-4 and backbone learning rates of 1e-5/3e-5 with AdamW. Keep frozen
  layers and their BatchNorm running statistics frozen for small batches.
- [ ] Use development-only early stopping (for example patience 5 within a
  maximum 30 epochs), restore the best checkpoint, and log train/validation
  loss and AP. More epochs alone are not evidence of improvement.
- [ ] Compare unweighted loss with weighted cross-entropy. Treat a balanced
  sampler or focal loss as separate optional experiments; do not combine all
  imbalance corrections by default. Evaluate resulting calibration too.
- [ ] Keep augmentation training-only. Start with the existing flips and mild
  affine transforms; review whether they preserve wound content. Test mild
  lighting variation separately. Avoid aggressive colour changes that remove
  possible redness information or crops that cut out the wound.
- [ ] Only after these baselines, compare one alternative pretrained backbone
  (e.g. EfficientNet-B0) on the same folds and budget. A larger model is not a
  guaranteed improvement on 686 images. Record inference cost as well.
- [ ] Do not assume Kaggle pretraining improves this task: the existing matched
  experiment reduced AUROC (0.725 to 0.695) and worsened Brier score despite a
  small AP increase. Any further transfer experiment needs matched controls.

## P3: Error analysis, segmentation and promotion

- [ ] Review false positives/negatives from development folds for background,
  lighting, dressings, wound type and label ambiguity. Avoid using test errors
  to design repeated improvements to the same reported test benchmark.
- [ ] Test whole-image versus wound-region input only after validating masks
  on the target dataset. Segmentation improvement does not automatically
  improve classification; poor masks can remove useful information.
- [ ] Report classifier-only scores on all selected images separately from
  service behavior. For quality-gated/deferred predictions report retake and
  defer rates, coverage, class-specific exclusions, and metrics on accepted
  cases. Never silently drop difficult images to increase reported accuracy.
- [ ] Save proposed weights under a new candidate filename with preprocessing,
  class order, calibration, threshold, split provenance and metrics. Do not
  overwrite `models/classifier.pt` or promote based on training accuracy.
- [ ] Promote only after the predeclared development criterion improves with
  consistent seed results, a locked independent evaluation is complete, and
  inference/checkpoint compatibility is verified. Document regressions and
  uncertainty; retain the old checkpoint for rollback.

## Suggested order for the next contributor

1. Complete P0 and reproduce the baseline.
2. Implement development-only objective/threshold comparisons from P1.
3. Run the frozen versus gradual-fine-tuning comparison from P2 in Colab.
4. Publish an experiment table including failures, choose once, and evaluate
   on the locked confirmation set. No higher score is promised in advance.

## Technical references

- [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html):
  tune on separate validation/CV data, not the data used to fit the classifier.
- [scikit-learn evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html):
  balanced accuracy and precision/recall measures complement ordinary accuracy
  on imbalanced data.
