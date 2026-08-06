# Dataset Notes

## Source
**NEU-CLS (Northeastern University Surface Defect Classification database)** — the same
1,800-image dataset commonly linked from the MVTec/Kaggle/Roboflow-style public sources listed
in the brief. The official NEU faculty homepage that originally hosted the dataset has been
intermittently unavailable for several years (a documented, widely-referenced problem — see
the GitHub mirrors below), so the dataset was retrieved from a public, unmodified GitHub mirror
that redistributes the original 1,800 images unchanged:

- Mirror used: `github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`
  (folders `IMAGES/` and `Validation_Images/`)
- Retrieved: 2026-08-06 (git clone, shallow, default branch, commit at time of clone)
- Original dataset citation: K. Song and Y. Yan, "A noise robust method based on completed
  local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*,
  2013 (Northeastern University Surface Inspection Laboratory).

## What one labeled example represents
Each image is a single **200×200 grayscale photograph** (stored as 3-channel JPG) of a small
patch of hot-rolled steel strip surface, cropped/captured so that exactly one of six defect
types is visible and centered. The label is image-level (one label per image) — there are no
bounding boxes or pixel masks associated with the classification-style version of the dataset,
only the six-way class folder/filename structure.

## Classes (6)
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`

## Image count
- 1,800 images total as distributed (300/class nominal)
- **1,799** after removing one exact duplicate found during inspection (see below)
- Image dimensions: uniformly 200×200 pixels, RGB-encoded JPG (content is grayscale)

## Label format
Label is encoded in the filename prefix, e.g. `crazing_1.jpg`, `pitted_surface_176.jpg`.
No XML/JSON/CSV label file is required for classification; the filename-derived class name
is parsed once at data-intake time and is **not** used as a model input feature.

## Missing / corrupt image check
Programmatically opened and `.verify()`-checked all 1,800 files: **0 corrupt files**, all
images load successfully at (200, 200) RGB.

## Duplicate / near-duplicate check
Computed an MD5 hash of every image file's raw bytes. Found **one exact byte-for-byte
duplicate pair**: `IMAGES/patches_101.jpg` and `IMAGES/patches_105.jpg`. One copy
(`patches_105.jpg`) was dropped from the pool before splitting so the duplicate could not be
placed in two different splits. No near-duplicate (perceptual-hash) check beyond exact-byte
matching was run; given the dataset's known provenance (curated lab captures, not scraped/
augmented web images) exact-hash duplication was judged the primary leakage risk.

## Scope decision: classification path, not detection
The source dataset also ships XML bounding-box annotations for a subset (in the mirror's
`ANNOTATIONS/` folder) that could support Path B (object/defect localization). This project
uses **Path A2 (Labeled Defect Classification)** instead: the annotations were not used, and
the bounding boxes were not converted to a detection task. This is a deliberate scope
decision to keep the project within a single classification pipeline, not a data limitation —
it is documented here so a reviewer knows Path B was available but intentionally not taken.

## Business framing / target-class decision
The raw dataset has **no "normal / defect-free" class** — every image already depicts one of
six known defect types (it was built for classifying *which* defect is present, not screening
"defective vs. not"). This project's business framing (see `vision_evaluation_report.md`)
therefore treats `crazing` as the safety-critical target defect and evaluates it with a
dedicated one-vs-rest threshold analysis on top of the full 6-class evaluation. This is called
out explicitly because it means the "false-negative cost" framing in this project is about
*missing the crazing label specifically*, not about missing defects vs. clean stock.
