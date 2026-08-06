# Qualitative Prediction Review

![qualitative panel](../artifacts/qualitative_panel.png)

11 real test images, model prediction and confidence shown above each:

- **FALSE NEG / MISS (4 panels, top row):** the 5 test misclassifications minus one repeated
  layout slot — see `false_negative_analysis.md` for the full table of all 5 with exact
  confidences. All four `inclusion` ⇄ `pitted_surface` confusions shown here look like faint,
  low-contrast streak/speckle textures even to visual inspection — a genuinely hard pair.
- **HARD (low-confidence) correct (2 panels):** cases the model got right but with prediction
  confidence near 0.5–0.83 — both are `inclusion` images with faint markings, i.e. the same
  visual ambiguity that produces the false negatives above, just landing on the correct side
  of the decision boundary this time.
- **EASY (high-confidence) correct (2 panels):** a `scratches` image (bright, high-contrast
  linear streaks) and a `patches` image (large, dark blob-like regions) — both visually
  distinctive defect signatures that the model calls at ~1.00 confidence.
- **Target-class (crazing) true positives (2 panels):** two `crazing` images called correctly
  at ~1.00 confidence — fine, hairline crack networks across the whole crop, visually distinct
  from the smoother `inclusion`/`pitted_surface` textures once you know what to look for.

## Takeaway
The qualitative panel corroborates the confusion-matrix pattern: **every real error involves
the inclusion/pitted_surface pair**, and the model's own confidence score is informative — the
two lowest-confidence correct predictions sit in the same visual territory as the four
confident misses, meaning a "flag anything under ~0.85 confidence for human review" rule would
catch most of this project's real error cases in addition to the target `crazing` false-alarm
case, at the cost of routing a modest number of easy correct calls to review as well.
