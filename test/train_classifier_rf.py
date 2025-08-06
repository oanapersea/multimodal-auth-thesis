import os, sqlite3, joblib, collections
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

DB_PATH = "auth.db"
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "face_rf_model.joblib"
DIM_FACE = 128
ZERO_DIV = 0
N_VAL_PER_USER = 2

conn = sqlite3.connect(DB_PATH)
rows = conn.execute("""
    SELECT f.id,          
           f.orig_id,   
           f.is_augmented,
           u.username,
           f.embedding
      FROM face_embeddings f
      JOIN users u ON u.id = f.user_id
""").fetchall()
conn.close()


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
        Xtr.append(v);
        ytr.append(user)
    elif key in val_pairs and is_aug == 0:
        Xvl.append(v);
        yvl.append(user)

if not Xtr or not Xvl:
    raise RuntimeError("Not enough data after splitting")

Xtr = np.stack(Xtr);
ytr = np.array(ytr)
Xvl = np.stack(Xvl);
yvl = np.array(yvl)

print("Train set:", collections.Counter(ytr))
print("Val set:", collections.Counter(yvl))

base_rf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
)
base_rf.fit(Xtr, ytr)

rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)
rf.fit(Xtr, ytr)
classes = list(rf.classes_)

print("\nValidation report:")
print(classification_report(yvl, rf.predict(Xvl),
                            labels=classes,
                            target_names=classes,
                            zero_division=ZERO_DIV))
print("Confusion matrix:\n",
      confusion_matrix(yvl, rf.predict(Xvl), labels=classes))

pvl = rf.predict_proba(Xvl)
class_thresholds = {}
for i, cls in enumerate(classes):
    bin_labels = (yvl == cls).astype(int)
    cls_scores = pvl[:, i]
    fpr_c, tpr_c, thr_c = roc_curve(bin_labels, cls_scores)
    eer_idx_c = np.nanargmin(np.abs((1 - tpr_c) - fpr_c))
    class_thresholds[cls] = float(thr_c[eer_idx_c])

genuine, impostor = [], []
for true, p in zip(yvl, pvl):
    i = classes.index(true)
    genuine.append(p[i])
    impostor.append(np.max(np.delete(p, i)))

labels_all = np.array([1] * len(genuine) + [0] * len(impostor))
scores_all = np.array(genuine + impostor)
fpr, tpr, thr = roc_curve(labels_all, scores_all)
eer_idx = np.nanargmin(np.abs((1 - tpr) - fpr))
global_threshold = float(thr[eer_idx])
print(f"\nRandomForest EER = {fpr[eer_idx]:.3f} | global_threshold = {global_threshold:.3f}")

joblib.dump({
    "lr": rf,
    "classes": classes,
    "global_threshold": global_threshold,
    "class_thresholds": class_thresholds
}, MODEL_FILE)
print("Saved random-forest model", MODEL_FILE)
