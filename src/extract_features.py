"""
Extract frozen pretrained-MobileNet (alpha=0.5, ImageNet, 128x128) embeddings
for every image in the train/val/test manifest.

Transfer-learning strategy: the ImageNet backbone is frozen (no weights are
updated). For the TRAIN split we additionally generate augmented copies
offline (flip / small rotation / brightness-contrast jitter) and embed each
copy, so the classifier head trained downstream sees augmented examples
without needing to re-run the CNN backbone every epoch (this keeps training
tractable on a single CPU core).
"""
import os
import io
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import tensorflow as tf

random.seed(42)
np.random.seed(42)

IMG_SIZE = 128
N_AUG_PER_TRAIN_IMAGE = 2  # number of ADDITIONAL augmented copies per train image (plus the original)

MANIFEST = "/home/claude/project/data/split_manifest.csv"
OUT_DIR = "/home/claude/project/data"
WEIGHTS = "/home/claude/mobilenet_a50_128_notop.h5"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- backbone: frozen ImageNet-pretrained MobileNet (alpha=0.5) ----
backbone = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=0.5, include_top=False,
    weights=None, pooling="avg"
)
backbone.load_weights(WEIGHTS)
backbone.trainable = False


def load_image(fp):
    im = Image.open(fp).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return im


def augment(im, rng):
    """Label-preserving augmentations appropriate for a steel-surface texture:
    the images have no canonical up/down orientation, so flips and small
    rotations do not change the defect identity. Brightness/contrast jitter
    approximates lighting variation on a real inspection line."""
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    angle = rng.uniform(-15, 15)
    im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    return im


def to_array_batch(images):
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in images], axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(arr)


def embed(images, batch_size=32):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = to_array_batch(images[i:i + batch_size])
        f = backbone.predict(batch, verbose=0)
        feats.append(f)
    return np.concatenate(feats, axis=0)


def main():
    df = pd.read_csv(MANIFEST)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(OUT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    for split in ["train", "val", "test"]:
        sub = df[df.split == split].reset_index(drop=True)
        images = []
        labels = []
        origin_filenames = []
        is_augmented = []
        rng = random.Random(42)

        for _, row in sub.iterrows():
            im = load_image(row.filepath)
            images.append(im)
            labels.append(class_to_idx[row.label])
            origin_filenames.append(row.filename)
            is_augmented.append(0)

            if split == "train":
                for _ in range(N_AUG_PER_TRAIN_IMAGE):
                    images.append(augment(im, rng))
                    labels.append(class_to_idx[row.label])
                    origin_filenames.append(row.filename)
                    is_augmented.append(1)

        feats = embed(images)
        labels = np.array(labels)
        np.save(os.path.join(OUT_DIR, f"{split}_features.npy"), feats)
        np.save(os.path.join(OUT_DIR, f"{split}_labels.npy"), labels)
        meta = pd.DataFrame({
            "filename": origin_filenames,
            "label_idx": labels,
            "label": [classes[i] for i in labels],
            "is_augmented": is_augmented,
        })
        meta.to_csv(os.path.join(OUT_DIR, f"{split}_meta.csv"), index=False)
        print(f"{split}: {feats.shape[0]} embedded rows "
              f"({sub.shape[0]} source images, "
              f"{feats.shape[0] - sub.shape[0]} augmented extras), "
              f"feature dim={feats.shape[1]}")


if __name__ == "__main__":
    main()
