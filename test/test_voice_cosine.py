import joblib, numpy as np
from scipy.spatial.distance import cosine
from db import get_audio_embeddings, get_all_usernames
import config

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine(a, b)

thr_dict = joblib.load(config.VOICE_MODEL_FILE).get("voice_thresholds", {})
DEFAULT_THR = 0.65

users         = get_all_usernames()
total_gen     = total_imp = 0
false_rejects = false_accepts = 0

for u in users:
    embs_u = get_audio_embeddings(u)
    if len(embs_u) < 2:
        continue

    t = thr_dict.get(u, DEFAULT_THR)

    for i in range(len(embs_u)):
        for j in range(i + 1, len(embs_u)):
            total_gen += 1
            if cos_sim(embs_u[i], embs_u[j]) < t:
                false_rejects += 1

    for v in users:
        if v == u:
            continue
        for emb_v in get_audio_embeddings(v):
            total_imp += 1
            if cos_sim(embs_u[0], emb_v) >= t:
                false_accepts += 1

FRR = false_rejects / total_gen if total_gen else 0.0
FAR = false_accepts / total_imp if total_imp else 0.0

print(f" Speakers              : {len(users)}")
print(f" Genuine pairs         : {total_gen:,}")
print(f" Impostor pairs        : {total_imp:,}")
print(f" False Rejects (FRR)   : {FRR*100:.2f} %")
print(f" False Accepts (FAR)   : {FAR*100:.2f} %")
