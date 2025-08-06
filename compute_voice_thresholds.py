import joblib
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import config
from config import VOICE_MODEL_FILE
from db import get_audio_embeddings, get_all_usernames

def compute_thresholds():
    voice_thresholds = {}
    users = get_all_usernames()
    for user in users:
        embs = get_audio_embeddings(user)
        if len(embs) < 2:
            print(f"Skipping {user}: need >2 samples, got {len(embs)}")
            continue

        genuine_sims = [
            1 - cosine(embs[i], embs[j])
            for i in range(len(embs))
            for j in range(i+1, len(embs))
        ]

        impostor_sims = []
        for other in users:
            if other == user:
                continue
            for emb2 in get_audio_embeddings(other):
                impostor_sims.append(1 - cosine(embs[0], emb2))

        y_true  = np.array([1]*len(genuine_sims) + [0]*len(impostor_sims))
        y_score = np.array(genuine_sims + impostor_sims)
        fpr, tpr, thr = roc_curve(y_true, y_score)
        idx = np.nanargmin(np.abs((1 - tpr) - fpr)) # finds the index where the difference between FAR and FRR is smallest
        voice_thresholds[user] = float(thr[idx]) #takes that threshold
        print(f"Threshold for {user}: {voice_thresholds[user]:.3f}")

    data = joblib.load(VOICE_MODEL_FILE)
    data["voice_thresholds"] = voice_thresholds
    joblib.dump(data, VOICE_MODEL_FILE)

    print(f"Updated {VOICE_MODEL_FILE} with per-user thresholds")

if __name__ == "__main__":
    compute_thresholds()
