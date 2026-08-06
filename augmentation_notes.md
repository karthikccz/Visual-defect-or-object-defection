# Augmentation Notes

Applied **only to the training split**, generating 2 additional augmented copies per training
image (so each of the 1,259 train images contributes 3 embedded rows: 1 original + 2
augmented → 3,777 total train rows). Validation and test images are never augmented.

| Augmentation | Range | Why it matches this visual problem |
|---|---|---|
| Horizontal flip | 50% chance | The steel-strip surface has no canonical left/right orientation — a crop from a moving strip looks equally valid mirrored. Does not change which defect type is present. |
| Vertical flip | 50% chance | Same reasoning — the strip's rolling direction in a single 200×200 crop is not a meaningful "up/down" semantic the model should key on. |
| Rotation | ±15° | Camera/strip alignment on a real line is not perfectly fixed; small rotations approximate that jitter without cropping the defect out of frame or creating an unrealistic viewpoint. |
| Brightness jitter | ×[0.85, 1.15] | Approximates lighting variation across the inspection line (fixtures, strip reflectivity) without washing out or inverting defect contrast. |
| Contrast jitter | ×[0.85, 1.15] | Same lighting-variation rationale as brightness. |

## What was deliberately excluded
- **No random crop / zoom.** These 200×200 images are already tight crops centered on the
  defect; cropping further risks cutting the defect out of frame entirely, which would make
  the label wrong (a labeling error, not a helpful augmentation).
- **No color jitter / channel shuffling.** The images are grayscale content stored as RGB;
  channel-level color augmentation has no physical meaning here and would only add noise.
  Brightness/contrast (applied uniformly across channels) is used instead.
- **No blur or heavy noise injection.** Some of the six defect types (`crazing` in particular)
  are already subtle, low-contrast hairline textures. Aggressive blur/noise risks erasing the
  very feature that distinguishes `crazing` from a clean patch, i.e. changing the effective
  label without changing the file's class tag — exactly the kind of label-changing
  augmentation the brief says to avoid.
