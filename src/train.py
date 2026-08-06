import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

tf.random.set_seed(42)
np.random.seed(42)

DATA = "/home/claude/project/data"
ART = "/home/claude/project/artifacts"
os.makedirs(ART, exist_ok=True)

classes = json.load(open(os.path.join(DATA, "classes.json")))
n_classes = len(classes)

Xtr = np.load(os.path.join(DATA, "train_features.npy"))
ytr = np.load(os.path.join(DATA, "train_labels.npy"))
Xval = np.load(os.path.join(DATA, "val_features.npy"))
yval = np.load(os.path.join(DATA, "val_labels.npy"))
Xtest = np.load(os.path.join(DATA, "test_features.npy"))
ytest = np.load(os.path.join(DATA, "test_labels.npy"))

# ---------------------------------------------------------------
# Baseline: majority-class reference, evaluated on val/test (not just shown)
# ---------------------------------------------------------------
majority_class = int(pd.Series(ytr).value_counts().idxmax())
base_val_pred = np.full_like(yval, majority_class)
base_test_pred = np.full_like(ytest, majority_class)
baseline = {
    "majority_class": classes[majority_class],
    "val_accuracy": float(accuracy_score(yval, base_val_pred)),
    "val_macro_f1": float(f1_score(yval, base_val_pred, average="macro", zero_division=0)),
    "test_accuracy": float(accuracy_score(ytest, base_test_pred)),
    "test_macro_f1": float(f1_score(ytest, base_test_pred, average="macro", zero_division=0)),
}
json.dump(baseline, open(os.path.join(ART, "baseline_results.json"), "w"), indent=2)
print("BASELINE:", baseline)

# ---------------------------------------------------------------
# Classifier head: small MLP on frozen 512-d MobileNet embeddings
# (this is the ONLY part with trainable weights -> the backbone stays frozen)
# ---------------------------------------------------------------
EPOCHS = 30
BATCH = 64
LR = 1e-3
SEED = 42

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_classes, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(Xtr, ytr, validation_data=(Xval, yval),
                     epochs=EPOCHS, batch_size=BATCH, verbose=2)

# per-epoch val macro-F1 (Keras doesn't track this natively) -> recompute
val_macro_f1_per_epoch = []
# retrain isn't needed; we saved weights each epoch via checkpoint callback instead.
# Simpler: redo fit with a callback that checkpoints on val macro-F1.

class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_f1 = -1
        self.best_epoch = -1
        self.best_weights = None
        self.log = []

    def on_epoch_end(self, epoch, logs=None):
        preds = np.argmax(self.model.predict(Xval, verbose=0), axis=1)
        f1 = f1_score(yval, preds, average="macro", zero_division=0)
        acc = accuracy_score(yval, preds)
        self.log.append({"epoch": epoch + 1, "val_loss": float(logs["loss"]),
                          "val_accuracy": float(acc), "val_macro_f1": float(f1)})
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_epoch = epoch + 1
            self.best_weights = [w.copy() for w in self.model.get_weights()]


# Re-run training from scratch with the checkpoint callback (keeps it simple/reproducible)
tf.random.set_seed(SEED)
model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_classes, activation="softmax"),
])
model2.compile(optimizer=tf.keras.optimizers.Adam(LR),
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

cb = MacroF1Checkpoint()
model2.fit(Xtr, ytr, validation_data=(Xval, yval),
           epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[cb])

model2.set_weights(cb.best_weights)
model2.save(os.path.join(ART, "classifier_head.keras"))

pd.DataFrame(cb.log).to_csv(os.path.join(ART, "training_curve.csv"), index=False)
print(f"Best checkpoint: epoch {cb.best_epoch}, val_macro_f1={cb.best_f1:.4f}")

config = {
    "backbone": "MobileNet (Keras, ImageNet-pretrained, alpha=0.5, 128x128 input, frozen)",
    "backbone_params": 829536,
    "head_architecture": "Dense(128, relu) -> Dropout(0.3) -> Dense(6, softmax)",
    "input_image_size": 128,
    "batch_size": BATCH,
    "epochs_run": EPOCHS,
    "learning_rate": LR,
    "optimizer": "Adam",
    "seed": SEED,
    "best_checkpoint_epoch": cb.best_epoch,
    "best_val_macro_f1": cb.best_f1,
    "selection_rule": "checkpoint with highest validation macro-F1 across training; test set never used for selection",
    "n_augmented_train_copies_per_image": 2,
}
json.dump(config, open(os.path.join(ART, "training_config.json"), "w"), indent=2)
print(json.dumps(config, indent=2))
