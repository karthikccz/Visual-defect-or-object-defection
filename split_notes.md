# Split Notes

- **No official train/val/test split was used.** The mirror's own `IMAGES` / `Validation_Images`
  folders are an arbitrary ~98%/2% split with no documented methodology, so this project pools
  all 1,799 (de-duplicated) images and creates its own split.
- **Method:** stratified split via `sklearn.model_selection.train_test_split`, two-stage
  (70% train / 15% val / 15% test), stratified on class label at each stage, `random_state=42`.
- **Reproducibility:** the exact split is written to `data/split_manifest.csv` (filepath,
  filename, label, split) and can be regenerated deterministically by re-running the split
  step with the same seed.
- **Leakage prevention:**
  - The one exact-duplicate image pair found during inspection was de-duplicated *before*
    splitting (see `dataset_notes.md`), so it cannot appear in two different splits.
  - Verified programmatically that `train`, `val`, and `test` file-path sets are pairwise
    disjoint (assertion in the split script).
  - All augmented copies are generated **after** the split, only from images already assigned
    to `train`, and are never added to `val` or `test` (see `augmentation_notes.md`).
- **Class representation preserved:** every class has (210 train / 45 val / 45 test), except
  `patches` (209 / 45 / 45) because it lost one image to de-duplication.
- **Final split sizes:** train = 1,259 source images, val = 270, test = 270.
