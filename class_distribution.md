# Class Distribution

Pooled dataset (post de-duplication): **1,799 images**, nominally balanced across 6 classes
(300/class before de-dup; `patches` has 299 after dropping the duplicate).

| class            | pooled count | train | val | test |
|------------------|-------------:|------:|----:|-----:|
| crazing          | 300          | 210   | 45  | 45   |
| inclusion        | 300          | 210   | 45  | 45   |
| patches          | 299          | 209   | 45  | 45   |
| pitted_surface   | 300          | 210   | 45  | 45   |
| rolled-in_scale  | 300          | 210   | 45  | 45   |
| scratches        | 300          | 210   | 45  | 45   |
| **total**        | **1,799**    | **1,259** | **270** | **270** |

![class distribution](../artifacts/class_distribution.png)

## Imbalance assessment
The dataset is **effectively balanced** (max/min class ratio = 300/299 ≈ 1.003). This is a
deliberate property of the NEU-CLS collection protocol (300 curated samples per defect type),
not a reflection of true production-line defect-rate imbalance. This matters for how the
results should be read: **a real inspection line will see crazing (and other defects) far
less often than 1-in-6 of all images** — most images on a real line are defect-free steel
with no anomaly at all. Because this public dataset only contains defect images (see
`dataset_notes.md`), the class-balance metrics reported here (accuracy, macro-F1, per-class
recall) describe how well the model tells six *known* defect types apart, not how it would
perform at realistic pilot-time defect prevalence (which would need a "clean stock" negative
class this dataset does not provide — see `vision_evaluation_report.md`, Limitations).

## Minority classes / label quality
No class is a meaningful minority (all within 1 image of 300). No missing labels — every file
matched exactly one of the six expected class prefixes. No ambiguous/multi-defect images were
flagged during the qualitative review pass (each image visually shows a texture consistent
with its filename label; see `qualitative_predictions.md`).

Because the classes are balanced, **accuracy is not automatically misleading here the way it
would be under real skew** — but it is still reported only as a secondary metric per the task
requirements, with macro-F1, per-class precision/recall, and a confusion matrix as primary
evidence, since those are what would need to keep working if/when this pipeline is retrained
on a production sample with genuine defect-rate imbalance.
