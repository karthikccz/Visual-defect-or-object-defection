# Vision Evaluation Report — Steel Surface Defect Classification

## 1. Business scenario
A hot-rolled steel strip inspection line wants automated screening support for six known
surface-defect types. Among these, **`crazing`** (fine hairline crack networks) is treated as
the safety-critical target: it is visually subtle and can be a precursor to strip failure
downstream, so a missed crazing detection is far more costly than a false alarm on a lower-
severity defect like `scratches` or `patches`. The target user is a line inspector or quality
engineer who would receive model flags as a second check alongside manual inspection, not as
an unsupervised pass/fail gate. Expected action after a flag: route the image (and, in a real
deployment, the physical coil location) for manual confirmation.

## 2. Task path and dataset
**Path A2 (Labeled Defect Classification)**, using NEU-CLS (1,799 images after de-duplication,
6 balanced classes, 200×200 grayscale). Full provenance, label format, class balance, and
split methodology are documented in `dataset_notes.md`, `label_format_notes.md`,
`class_distribution.md`, and `split_notes.md`.

**Important framing caveat:** this public dataset contains only defect images — there is no
"clean stock" negative class. All metrics below describe how well the model tells six known
defect types apart from each other, not how it would perform screening real production images
where most frames are defect-free (see Limitations).

## 3. Model summary
A frozen ImageNet-pretrained MobileNet (alpha=0.5, 128×128 input, 829,536 params, 5.4MB) is
used as a fixed feature extractor, with a small trainable classifier head
(`Dense(128,relu)→Dropout(0.3)→Dense(6,softmax)`, 804KB) trained on the cached 512-d
embeddings. This keeps the whole pipeline runnable on a single CPU core with no GPU (feature
extraction: ~38s for the full dataset including augmented copies; head training: seconds).
Full configuration: `training_config.md`.

## 4. Augmentation rationale
Flips, ±15° rotation, and brightness/contrast jitter — chosen because the texture has no
canonical orientation and lighting varies on a real line. Crop/zoom and blur/noise were
deliberately excluded to avoid erasing the subtle `crazing` signal. Full rationale:
`augmentation_notes.md`.

## 5. Metric results

| | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline (val) | 0.1667 | 0.0476 |
| Majority-class baseline (test) | 0.1667 | 0.0476 |
| Model (test) | **0.9815** | **0.9814** |

Per-class test precision/recall (support 45 each):

| class | precision | recall | f1 |
|---|---:|---:|---:|
| crazing | 0.978 | 1.000 | 0.989 |
| inclusion | 0.977 | 0.933 | 0.955 |
| patches | 1.000 | 1.000 | 1.000 |
| pitted_surface | 0.935 | 0.956 | 0.945 |
| rolled-in_scale | 1.000 | 1.000 | 1.000 |
| scratches | 1.000 | 1.000 | 1.000 |

Crazing one-vs-rest ranking quality: ROC-AUC = 1.000, PR-AUC = 1.000. Full numbers:
`metric_results.csv`, `artifacts/test_metrics.json`.

The model clears the baseline by a wide margin (16.7% → 98.15% accuracy), and — more
importantly given the evaluation standard for this project — clears it on macro-F1 and
per-class recall as well, not just aggregate accuracy.

## 6. Threshold comparison
Compared `P(crazing) ≥ 0.5` (recall 1.000, precision 0.978) against a conservative
`P(crazing) ≥ 0.99` (recall 0.800, precision 1.000). Recommendation: keep the low threshold —
the false-alarm cost this test set actually exhibits (1 extra review) is cheap relative to a
missed crazing detection. Full writeup: `threshold_comparison.md`.

## 7. False-negative findings
5 real test misclassifications (of 270), all concentrated in one visually-ambiguous pair
(`inclusion` ⇄ `pitted_surface`), plus one `pitted_surface`→`crazing` false alarm. **Zero
false negatives on the crazing class itself** at the default threshold. Two of the five errors
are near-50/50, low-confidence calls rather than confident mistakes — a favorable failure
mode, since a confidence-based review rule would catch them. Full table and per-example
hypotheses: `false_negative_analysis.md`, `decision_log.md` (Q13–14).

## 8. Qualitative examples
See `qualitative_predictions.md` and `artifacts/qualitative_panel.png` — 11 real test images
covering all 5 misclassifications, the 2 lowest-confidence correct predictions, the 2
highest-confidence correct predictions, and 2 crazing true positives.

## 9. Resource and deployment constraints
- **Total model size:** ~6.2MB (5.4MB frozen backbone + 0.8MB head) — small enough to ship as
  part of an edge/on-line inspection client, not just a server-side service.
- **Measured inference latency (single CPU core, unbatched, no GPU):** ~53ms/image through the
  frozen backbone + ~50ms/image of `predict()` call overhead for the tiny head (dominated by
  per-call framework overhead, not actual head compute, since the head is a 512→128→6 MLP).
  Real deployments should **batch requests** rather than call `predict()` per image — the head
  computation itself is negligible, so batched throughput would be far higher than the ~100ms/
  image serial figure suggests.
- **GPU/CPU assumption:** none required for this pipeline as built; a GPU would mainly help if
  the backbone were fine-tuned end-to-end rather than used frozen (not done here — see
  Limitations).
- **Preprocessing requirement:** resize to 128×128, MobileNet-style `preprocess_input` scaling
  to [-1, 1] — cheap, standard image-pipeline work.
- **Monitoring signal:** track the fraction of production predictions with top-class
  confidence below ~0.85 (see `decision_log.md`, Q16) as a drift indicator.

## 10. Limitations
- **No true negative/normal class in the source data.** All reported metrics measure
  distinguishing six known defect types from each other, not detecting "defect vs. clean
  stock." A production pilot needs a labeled sample of defect-free strip images before these
  numbers can be read as production-readiness evidence for a screening (not just
  classification) use case.
- **Frozen backbone, not fine-tuned.** The ImageNet backbone's filters were never adapted to
  this specific steel-texture domain; the strong result here comes from a lightweight
  classifier head on top of generic ImageNet features. Fine-tuning (with more compute) might
  close the remaining inclusion/pitted_surface confusion, or might not — untested here.
  Rerunning with a defrosted final backbone block would be the direct next hardware-cost
  trade-off to evaluate.
- **Curated benchmark, not production imagery.** NEU-CLS images are lab-quality, centered,
  single-defect crops. Real line imagery will include partial defects, motion blur, occlusion,
  multiple simultaneous defects per frame, and far lower defect prevalence than this balanced
  benchmark. The near-perfect scores here (crazing ROC-AUC = 1.000) should be read as "this
  method works on this benchmark," not as evidence of production readiness.
- **Small held-out test set (45 crazing images).** The 0-false-negative crazing result is
  genuine but based on a modest sample; it should not be treated as a guarantee that recall
  stays at 100% on a larger or more varied production sample.

## 11. Recommended next steps
1. Collect (or synthesize via controlled negative sampling) a defect-free "clean stock"
   image set so the pilot can be evaluated on the actual screening task (defect vs. no defect),
   not just defect-type classification.
2. Pilot the epoch-3 checkpoint at the default threshold on a small batch of real production
   images before any wider rollout, specifically watching the confidence-distribution
   monitoring signal described above.
3. If inclusion/pitted_surface confusion matters operationally (i.e., if those two defect
   types trigger different downstream actions), evaluate fine-tuning the last backbone block
   with the extra compute budget that would require, rather than only training the head.
