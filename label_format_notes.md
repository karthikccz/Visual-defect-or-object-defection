# Label Format Notes

- **Format:** filename-encoded class label, e.g. `<class>_<index>.jpg` (`crazing_14.jpg`,
  `rolled-in_scale_38.jpg`). No separate label file for the classification task.
- **Parsing rule:** class = filename with the trailing `_<digits>.<ext>` stripped, lower-cased.
  Implemented once in `src/data prep` (see split manifest generation) via a single regex
  applied at intake time.
- **Prediction-time availability constraint compliance:** the parsed class name is used
  *only* to build the train/val/test manifest and evaluate predictions. The filename itself
  is never passed to the model — the model only ever sees the 200×200 pixel image (resized to
  128×128 and normalized), so no label-derived metadata leaks into the feature pipeline.
- **Detection-style annotations (unused):** the source mirror also includes an
  `ANNOTATIONS/` folder with Pascal-VOC-style XML files (bounding box + class per image) for
  the same images. These were **not used** in this project (see `dataset_notes.md` —
  Path A2 classification was chosen over Path B detection). They are noted here for
  completeness in case a future iteration extends this project to Path B.
