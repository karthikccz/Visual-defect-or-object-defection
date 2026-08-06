import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score,
                              average_precision_score)

DATA = "/home/claude/project/data"
ART = "/home/claude/project/artifacts"

classes = json.load(open(os.path.join(DATA, "classes.json")))
crazing_idx = classes.index("crazing")

Xtest = np.load(os.path.join(DATA, "test_features.npy"))
ytest = np.load(os.path.join(DATA, "test_labels.npy"))
meta = pd.read_csv(os.path.join(DATA, "test_meta.csv"))
manifest = pd.read_csv(os.path.join(DATA, "split_manifest.csv"))
fp_lookup = {r.filename: r.filepath for r in manifest.itertuples()}

model = tf.keras.models.load_model(os.path.join(ART, "classifier_head.keras"))
probs = model.predict(Xtest, verbose=0)
preds = np.argmax(probs, axis=1)

# ---- full multiclass metrics ----
report = classification_report(ytest, preds, target_names=classes,
                                digits=4, output_dict=True)
cm = confusion_matrix(ytest, preds)
metrics = {
    "test_accuracy": float(accuracy_score(ytest, preds)),
    "test_macro_f1": float(f1_score(ytest, preds, average="macro")),
    "per_class": report,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_labels": classes,
}
json.dump(metrics, open(os.path.join(ART, "test_metrics.json"), "w"), indent=2)

pd.DataFrame(report).T.to_csv(os.path.join(ART, "..", "metric_results.csv"))

# ---- crazing one-vs-rest PR-AUC / ROC-AUC ----
p_crazing = probs[:, crazing_idx]
y_bin = (ytest == crazing_idx).astype(int)
crazing_rocauc = roc_auc_score(y_bin, p_crazing)
crazing_prauc = average_precision_score(y_bin, p_crazing)

# ---- threshold comparison on the crazing (safety-critical) one-vs-rest score ----
thresholds = [0.5, 0.99]
thr_rows = []
for thr in thresholds:
    pb = (p_crazing >= thr).astype(int)
    tp = int(((pb == 1) & (y_bin == 1)).sum())
    fp = int(((pb == 1) & (y_bin == 0)).sum())
    fn = int(((pb == 0) & (y_bin == 1)).sum())
    tn = int(((pb == 0) & (y_bin == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    thr_rows.append({"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                      "precision": prec, "recall": rec})
thr_df = pd.DataFrame(thr_rows)
thr_df.to_csv(os.path.join(ART, "threshold_table.csv"), index=False)

json.dump({"crazing_roc_auc": crazing_rocauc, "crazing_pr_auc": crazing_prauc},
          open(os.path.join(ART, "crazing_ovr_auc.json"), "w"), indent=2)

# ---- false negatives (every test misclassification) ----
mism_idx = np.where(preds != ytest)[0]
fn_rows = []
for i in mism_idx:
    fn_rows.append({
        "filename": meta.loc[i, "filename"],
        "true_label": classes[ytest[i]],
        "predicted_label": classes[preds[i]],
        "predicted_confidence": float(probs[i, preds[i]]),
        "true_class_confidence": float(probs[i, ytest[i]]),
    })
pd.DataFrame(fn_rows).to_csv(os.path.join(ART, "false_negatives_raw.csv"), index=False)

print("Test accuracy:", metrics["test_accuracy"])
print("Test macro F1:", metrics["test_macro_f1"])
print("Crazing one-vs-rest ROC-AUC:", crazing_rocauc, "PR-AUC:", crazing_prauc)
print(thr_df)
print(f"{len(mism_idx)} misclassifications on {len(ytest)} test images")
