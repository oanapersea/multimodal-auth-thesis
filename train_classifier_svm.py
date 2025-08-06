import os, sqlite3, joblib, collections
import numpy as np
from pathlib import Path
from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm           import SVC
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.metrics       import classification_report, confusion_matrix, roc_curve
import db
import config

DB_PATH     = config.DB_PATH
MODEL_DIR   = config.MODELS_DIR
MODEL_FILE  = config.FACE_MODEL_FILE
DIM_FACE    = 128
ZERO_DIV    = 0
N_VAL_PER_USER = 2

rows = db.get_all_face_rows()

def decode(blob, dim=DIM_FACE):
    v = np.frombuffer(blob, dtype=np.float64)
    if v.size != dim or np.any(~np.isfinite(v)):
        return None
    return v.astype(np.float32)

from collections import defaultdict
user_to_origs = defaultdict(list)
for _, orig_id, is_aug, user, _ in rows:
    if is_aug == 0:
        user_to_origs[user].append(orig_id)

val_pairs = set()
train_pairs = set()
for user, origs in user_to_origs.items():
    origs = list(set(origs))
    np.random.shuffle(origs)
    val_chosen = origs[:N_VAL_PER_USER]
    train_chosen = origs[N_VAL_PER_USER:]
    val_pairs.update((oid, user) for oid in val_chosen)
    train_pairs.update((oid, user) for oid in train_chosen)

Xtr, ytr, Xvl, yvl = [], [], [], []
for _, orig_id, is_aug, user, blob in rows:
    v = decode(blob)
    if v is None:
        continue
    key = (orig_id, user)
    if key in train_pairs:
        Xtr.append(v); ytr.append(user)
    elif key in val_pairs and is_aug == 0:
        Xvl.append(v); yvl.append(user)

if not Xtr or not Xvl:
    raise RuntimeError("Not enough data after splitting!")

#SVM expects a matrix and an array
Xtr = np.stack(Xtr); ytr = np.array(ytr)
Xvl = np.stack(Xvl); yvl = np.array(yvl)

print("Train set:", collections.Counter(ytr))
print("Val set:", collections.Counter(yvl))


base_svm = make_pipeline(
        StandardScaler(with_mean=False),
        SVC(kernel="linear",
            probability=False,
            class_weight="balanced",
            random_state=42)
)

svm = CalibratedClassifierCV(base_svm, method="isotonic", cv=3)
svm.fit(Xtr, ytr)

classes = list(svm.classes_)

print("\nValidation report:")
print(classification_report(yvl, svm.predict(Xvl),
                            labels=classes,
                            target_names=classes,
                            zero_division=ZERO_DIV))
print("Confusion matrix:\n",
      confusion_matrix(yvl, svm.predict(Xvl), labels=classes))

pvl = svm.predict_proba(Xvl)
genuine, impostor = [], []
for true, p in zip(yvl, pvl):
    i = classes.index(true)
    genuine.append(p[i])
    impostor.append(np.max(np.delete(p, i)))

labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
scores = np.concatenate([genuine, impostor])
fpr, tpr, thr = roc_curve(labels, scores)
eer_idx  = np.nanargmin(np.abs((1 - tpr) - fpr))
best_thr = thr[eer_idx]
print(f"\nEqual-Error Rate = {fpr[eer_idx]:.3f}  |  threshold = {best_thr:.3f}")

class_thresholds = {}
for i, cls in enumerate(classes):
    bin_labels = (yvl == cls).astype(int) # binary vector for the current user
    cls_scores = pvl[:, i]   # get all prediction probabilities for the user
    fpr_c, tpr_c, thr_c = roc_curve(bin_labels, cls_scores)
    eer_idx_c = np.nanargmin(np.abs((1 - tpr_c) - fpr_c))
    computed_threshold = float(thr_c[eer_idx_c])
    capped_threshold = min(0.95, computed_threshold)
    class_thresholds[cls] = capped_threshold

print("\nPer-user (EER) thresholds")
for cls, thr_v in sorted(class_thresholds.items()):
    print(f"{cls:<15s}: {thr_v:.3f}")


joblib.dump({
    "svm": svm,
    "classes": classes,
    "global_threshold": best_thr,
    "class_thresholds": class_thresholds
}, MODEL_FILE)

print("Saved model with per-class thresholds →", MODEL_FILE)



