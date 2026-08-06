# Threshold Comparison

Full 6-class predictions use `argmax` over the softmax output (equivalent to a per-class
threshold of ~1/6 in a relative sense). For the safety-critical target defect (`crazing`,
see business framing in `vision_evaluation_report.md`), this project additionally evaluates a
**one-vs-rest decision threshold on `P(crazing)`**, since that is the score a real "flag this
for inspection" alert would be built on.

## Two thresholds compared (test set, n=270, 45 true `crazing` images)

| Threshold | TP | FP | FN | TN | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 (default / argmax-equivalent) | 45 | 1 | 0 | 224 | 0.978 | **1.000** |
| 0.99 (conservative / high-confidence-only) | 36 | 0 | 9 | 225 | **1.000** | 0.800 |

One-vs-rest ranking quality (threshold-independent): ROC-AUC = 1.000, PR-AUC = 1.000 — the
`crazing` score is a clean separator on this test set (lowest true-positive score 0.955 vs.
highest true-negative score 0.668), which is why a threshold has to be pushed unusually high
(0.99) before it changes the outcome at all.

## Business interpretation
- **At the default threshold (0.5):** every true `crazing` image in the test set is caught
  (recall = 100%), at the cost of one false alarm — a `pitted_surface` image scored 0.668 for
  crazing and would be routed to the crazing-review queue unnecessarily. For a safety-critical
  defect where a missed detection can mean a downstream strip failure, one extra manual review
  per ~270 images is a reasonable trade.
- **At a stricter threshold (0.99):** the one false alarm disappears (precision reaches 100%),
  but **9 of the 45 true crazing images (20%) become false negatives** — they score between
  0.955 and 0.990, confident but not confident *enough* to clear the bar. Raising the bar this
  far buys nothing on precision that the pilot actually needs (only 1 false alarm existed to
  begin with) while giving up a fifth of true detections on the highest-consequence defect.
- **Recommendation:** keep the threshold low (0.5, i.e. plain argmax routing) for `crazing`
  specifically. The cost asymmetry here is clear — an extra manual review of a false alarm is
  cheap; a missed crazing crack that reaches the next process step is not. This recommendation
  should be revisited once the model is evaluated on genuine production data (see Limitations
  in `vision_evaluation_report.md`) rather than this balanced, curated benchmark, where scores
  are unusually well separated.
