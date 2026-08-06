# Decision Log

**1. Which dataset and task path did you choose?**
NEU-CLS (Northeastern University steel-surface defect classification dataset, 6 classes),
retrieved from a public GitHub mirror (`siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`)
since the original NEU faculty host is down. Task Path A2 — Labeled Defect Classification.

**2. How many images and classes were used?**
1,799 images (1,800 minus one exact duplicate found during inspection), across 6 classes:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches. Split into
1,259 train / 270 val / 270 test.

**3. What label format did the dataset use?**
Filename-encoded class prefix (e.g. `crazing_14.jpg`); no separate label file. Parsed once at
intake via regex and never used as a model input — only pixel data is fed to the model.

**4. What train/validation/test split did you use?**
Stratified 70/15/15 split via `sklearn.train_test_split` (two-stage), `random_state=42`.
Verified zero filepath overlap across the three splits. Train: 1,259, val: 270, test: 270,
class-balanced within ±1 image per class per split.

**5. If you used MVTec or another one-class dataset, how did you keep defect images out of
training and model-selection decisions?**
Not applicable — NEU-CLS is a labeled multi-class defect dataset (Path A2), not a one-class
anomaly-detection dataset (Path A1). There is no "normal/good" training-only class here; every
image in every split is a defect image, and each split's images are used exactly as normal
supervised-learning practice dictates (train for gradient updates, val for checkpoint
selection, test for final reporting only).

**6. What baseline or reference did you evaluate?**
Majority-class baseline (always predicts the most frequent training class, `crazing`),
evaluated on both val and test: **val accuracy = 0.1667, val macro-F1 = 0.0476; test accuracy
= 0.1667, test macro-F1 = 0.0476** — consistent with a roughly-balanced 6-class problem
(1/6 ≈ 0.167).

**7. What model architecture or checkpoint did you use?**
Frozen ImageNet-pretrained MobileNet (Keras/TensorFlow, width multiplier alpha=0.5, 128×128
input, `include_top=False`, global-average-pooled 512-d output, 829,536 frozen params) as a
feature extractor, feeding a trainable classifier head: `Dense(128, relu) → Dropout(0.3) →
Dense(6, softmax)`.

**8. What image size, batch size, epochs, and learning rate did you use?**
Image size 128×128 (resized from source 200×200). Batch size 64. 30 epochs run (full curve
logged). Learning rate 1e-3 with Adam. Seed 42.

**9. What augmentations did you apply and why?**
Horizontal flip, vertical flip, ±15° rotation, brightness jitter (±15%), contrast jitter
(±15%) — applied only to the training split, generating 2 extra augmented copies per training
image. Chosen because the steel-surface texture has no canonical orientation (flips/rotation
are label-preserving) and lighting varies on a real inspection line (brightness/contrast
jitter). Crop/zoom and blur/noise were deliberately excluded — see `augmentation_notes.md` for
the reasoning (risk of cutting out or erasing the very texture that defines the label,
especially for the subtle `crazing` class).

**10. What primary metric did you report, and why was accuracy not enough?**
Primary metrics: macro-F1, per-class precision/recall/F1, and confusion matrix, plus a
dedicated one-vs-rest ROC-AUC/PR-AUC and threshold sweep for the safety-critical `crazing`
class. Accuracy is reported only as a secondary figure. Even though this particular dataset
happens to be class-balanced (so accuracy isn't as distorting here as it would be under real
production defect-rate skew), the brief's evaluation standard is to never rely on accuracy
alone, and per-class/threshold metrics are what will actually matter once this pipeline meets
a real, imbalanced production stream.

**11. What were your main validation or test metric values?**
Best checkpoint (epoch 3): **validation macro-F1 = 1.0000**. Final test-set evaluation of that
checkpoint: **test accuracy = 0.9815, test macro-F1 = 0.9814**. Per-class test precision/recall
ranged from 0.935–1.000 / 0.933–1.000 (weakest class: `pitted_surface`, recall 0.956).
Crazing one-vs-rest: ROC-AUC = 1.000, PR-AUC = 1.000.

**12. Which two thresholds did you compare, and what changed in precision and recall?**
Compared `P(crazing) ≥ 0.5` vs. `P(crazing) ≥ 0.99` (one-vs-rest, test set). At 0.5:
precision = 0.978, recall = 1.000 (1 false positive, 0 false negatives). At 0.99:
precision = 1.000, recall = 0.800 (0 false positives, 9 false negatives). Raising the
threshold traded 20 recall points for a precision gain that wasn't actually needed (only one
false alarm existed at the lower threshold).

**13. How many false negatives did you find?**
5 total test misclassifications (each a false negative for its true class): 4 involve
`inclusion` ⇄ `pitted_surface` confusion, 1 involves a `pitted_surface` image scored highest
for `crazing` (a false alarm for the target class, not a miss). **Zero false negatives for the
`crazing` class specifically** at the default threshold.

**14. Paste or list 3 false-negative examples with image IDs and your hypothesis for each.**
- `inclusion_6.jpg` — true: inclusion, predicted: pitted_surface (conf 0.952). Hypothesis:
  faint, low-contrast streak texture visually closer to the pitted_surface class prototype.
- `inclusion_240.jpg` — true: inclusion, predicted: pitted_surface (conf 0.856). Hypothesis:
  same inclusion/pitted_surface visual overlap; confidently wrong, suggesting the two classes'
  feature-space centroids are close for this embedding.
- `pitted_surface_176.jpg` — true: pitted_surface, predicted: crazing (conf 0.668, moderate
  confidence). Hypothesis: some fine surface pitting can visually resemble faint crack
  networks at this resolution; this is a false alarm for the target class rather than a miss,
  which is the lower-risk failure direction given the stated business framing.

**15. Which threshold or checkpoint would you hand off for a pilot, and why?**
Hand off the **epoch-3 checkpoint at the default (argmax / 0.5) decision threshold**. It
maximizes validation macro-F1, generalizes to 98.15% test accuracy with errors concentrated in
one visually-ambiguous class pair, and — critically for the stated business risk — achieves
100% recall on the safety-critical `crazing` class with only one tolerable false alarm.
Raising the crazing threshold to 0.99 was evaluated and rejected: it sacrifices real
detections for a precision gain the pilot doesn't need at this false-alarm rate.

**16. What is one deployment constraint or monitoring signal you would track next?**
**Monitor the model's own confidence distribution on incoming production images, especially
the fraction scoring below ~0.85 on their top class.** This project's real errors and its
hardest-but-correct predictions both cluster in that low-confidence band (see
`false_negative_analysis.md` / `qualitative_predictions.md`), so a rising rate of
low-confidence predictions in production is a leading indicator that the model is drifting
away from the curated NEU-CLS distribution (different lighting, camera, or steel stock) before
accuracy metrics would show it — and would be the trigger for scheduling a re-training or
re-collection pass.
