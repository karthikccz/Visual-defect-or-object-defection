# Training / Model Configuration

## Transfer-learning strategy
Rather than fine-tuning the CNN backbone end-to-end (impractical on the available single-CPU-
core, no-GPU environment — see `README.md`), this project uses the standard **feature-extraction**
form of transfer learning:

1. A **frozen**, ImageNet-pretrained MobileNet (Keras/TensorFlow, `alpha=0.5`, 128×128 input,
   `include_top=False`, global-average-pooled 512-d output) is used purely as a fixed feature
   extractor. None of its 829,536 parameters are updated.
2. A small trainable **classifier head** is trained on top of the cached 512-d embeddings.

## Backbone
- Architecture: MobileNet v1, width multiplier `alpha=0.5`
- Pretraining: ImageNet-1k
- Weights source: `github.com/fchollet/deep-learning-models` release `v0.6`
  (`mobilenet_5_0_128_tf_no_top.h5`) — the original public release location for the exact
  weights later vendored into `keras.applications.MobileNet`
- Input size: 128×128×3 (resized from the source 200×200; `preprocess_input` scaling to [-1, 1])
- Params: 829,536 (frozen)
- File size: 5.4 MB (`.h5`)

## Classifier head
- Architecture: `Dense(128, relu) -> Dropout(0.3) -> Dense(6, softmax)`
- Params: ~68k (trainable)
- File size: 804 KB (`.keras`)
- Optimizer: Adam, learning rate = 1e-3
- Loss: sparse categorical cross-entropy
- Batch size: 64
- Epochs run: 30 (full curve logged, see `training_curve.csv`)
- Random seed: 42 (backbone loading, augmentation RNG, and `tf.random.set_seed`)

## Checkpoint / model selection
- A checkpoint callback recomputes **validation macro-F1 after every epoch** and keeps the
  weights from whichever epoch had the highest validation macro-F1.
- Best checkpoint: **epoch 3** (val_macro_f1 = 1.0000).
- The **test set was never used** for checkpoint selection, threshold selection, or any
  hyperparameter choice — it is touched only in the final evaluation step (`src/evaluate.py`).
- Full per-epoch validation loss/accuracy/macro-F1 log: `artifacts/training_curve.csv`.

## Why this fits free/limited compute
- Because the backbone is frozen, the (compute-heavy) forward pass through the CNN is run
  **once** per image (plus 2 augmented copies for train images) rather than once per epoch —
  embedding extraction for the full dataset (3,777 train rows incl. augmentations + 270 val +
  270 test) took **~38 seconds** on a single CPU core.
- Training the classifier head itself (30 epochs over cached 512-d vectors) took a few seconds
  total, since no CNN computation is involved.
- This makes the whole pipeline runnable on a laptop CPU, Colab free tier, or Kaggle free CPU
  quota with no GPU required, at the cost of not fine-tuning the backbone's own filters to this
  specific texture domain (see `vision_evaluation_report.md`, Limitations).
