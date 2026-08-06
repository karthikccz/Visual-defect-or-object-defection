# False-Negative Analysis

The test set produced **5 misclassifications out of 270 images** (test accuracy 98.15%). Per
class, this means: for whichever class was the true label, the model failed to flag it as
that class — i.e., each row below is a genuine false negative for its true class (and
simultaneously a false positive for whatever it was predicted as instead). Fewer than 10 exist,
so **all 5 are listed** below, per the brief's instruction for when the true count is small.

| # | filename | true label | predicted label | confidence in wrong label | confidence in true label | risk read |
|---|---|---|---|---:|---:|---|
| 1 | `inclusion_6.jpg` | inclusion | pitted_surface | 0.952 | 0.042 | High — model is confidently wrong |
| 2 | `inclusion_228.jpg` | inclusion | pitted_surface | 0.503 | 0.360 | Low — near-toss-up, borderline case |
| 3 | `pitted_surface_176.jpg` | pitted_surface | **crazing** | 0.668 | 0.176 | Medium — over-triggers the safety-critical class (a false alarm, not a miss, for crazing) |
| 4 | `inclusion_240.jpg` | inclusion | pitted_surface | 0.856 | 0.142 | High — confidently wrong |
| 5 | `pitted_surface_201.jpg` | pitted_surface | inclusion | 0.491 | 0.476 | Low — near-toss-up |

See `qualitative_predictions.md` / `artifacts/qualitative_panel.png` for the actual images.

## Pattern and likely reason for the misses
**4 of 5 errors are confusions between `inclusion` and `pitted_surface`.** Visually (see the
panel image), both classes present as faint, low-contrast speckled or streaked grayscale
texture on an otherwise uniform strip background — the two classes are the most visually
similar pair in the six-class set, and the errors go in both directions (inclusion→pitted, and
pitted→inclusion), consistent with genuine texture ambiguity rather than a directional model
bias. Two of the five errors (`inclusion_228`, `pitted_surface_201`) are also **low-confidence,
near 50/50 calls** — the model itself is signaling uncertainty rather than being confidently
wrong, which is the more benign failure mode (a downstream confidence threshold or
human-review-on-low-confidence rule would catch these).

## Business-critical class specifically (crazing)
There were **zero false negatives for `crazing`** in this test set (recall = 100%, see
`threshold_comparison.md`) — the one crazing-adjacent error (#3 above) is a *false positive*
(a `pitted_surface` image scored highest for crazing), which is the safer direction for the
stated business risk (missing a real crazing crack is worse than reviewing an extra false
alarm). This is a genuinely good result for the target defect, but it should be read
cautiously: this benchmark's `crazing` examples are curated, textbook-clear hairline-crack
crops. A real inspection line will show subtler, partially-occluded, or motion-blurred crazing
that this evaluation cannot speak to — see Limitations in `vision_evaluation_report.md`.
