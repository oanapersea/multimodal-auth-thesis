import os, glob, cv2
import numpy as np
import face_recognition
import albumentations as A
import sys

import config

DATA_DIR      = config.PROC_FACE_DIR
OUT_DIR       = config.AUG_FACE_DIR
N_AUG         = config.N_AUG
MAX_TRIES     = config.MAX_TRIES
LOW_SIM       = config.LOW_SIM
HIGH_SIM      = config.HIGH_SIM
FALLBACK_KEEP = 2

if len(sys.argv) > 1:
    user = sys.argv[1]
    USERS = [user]
else:
    USERS = sorted(os.listdir(DATA_DIR))

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


aug = A.Compose([
    A.Affine(rotate=(-15, 15),
             translate_percent={"x": (-0.10,0.10),"y": (-0.10,0.10)},
             scale=(0.9, 1.1), p=0.9),
    A.OneOf([
        A.OpticalDistortion(distort_limit=0.05),   # mimics lens distortion (lines and shapes shifted in non-linear way)
        A.GridDistortion(num_steps=5, distort_limit=0.05),   # each grid point randomly
        A.ElasticTransform(alpha=1, sigma=50),  # simulate elastic movements (facial muscle shifts)
    ], p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3,7)),
        A.MotionBlur(blur_limit=5),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20,
                         sat_shift_limit=30,
                         val_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80,140), p=0.4),   # brightens or darkens midtones
    A.GaussNoise(std_range=(10/255, 50/255), mean_range=(0.0, 0.0), p=0.4),  # pixel-level noise
    A.CLAHE(clip_limit=2.0, p=0.3),   # enhances local contrast in small regions of the image (features more prominent)
], p=1.0)


os.makedirs(OUT_DIR, exist_ok=True)

for user in USERS:
    src = os.path.join(DATA_DIR, user)
    dst = os.path.join(OUT_DIR, user)

    if not os.path.isdir(src):
        print(f"[WARNING] Source directory for user '{user}' does not exist. Skipping.")
        continue
    os.makedirs(dst, exist_ok=True)

    img_list = glob.glob(f"{src}/*.jpg")
    print(f"[DEBUG] Found {len(img_list)} images for user '{user}'.")

    for img_path in img_list:
        print(f"\n[DEBUG] Original image: {img_path}")
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[ERROR] Failed to load image: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb)
        if not locs:
            print(f"[WARNING] No face in {user}/{os.path.basename(img_path)}, skipping")
            continue

        emb_o = face_recognition.face_encodings(rgb, known_face_locations=[locs[0]])[0]

        kept_imgs = []
        kept_sims = []
        tries = 0
        while (len(kept_imgs) < N_AUG) and (tries < MAX_TRIES):
            tries += 1
            aug_bgr = aug(image=bgr)["image"]
            aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)

            locs2 = face_recognition.face_locations(aug_rgb)
            if not locs2:
                print(f"[DEBUG] [SKIP] augment#{tries} for {img_path} (no face detected in augmented image)")
                continue

            emb_a = face_recognition.face_encodings(aug_rgb, known_face_locations=[locs2[0]])[0]
            sim = cos_sim(emb_o, emb_a)
            print(f"[DEBUG] augment#{tries}: cosine sim={sim:.3f}")

            kept_sims.append((sim, aug_bgr))
            if LOW_SIM <= sim <= HIGH_SIM:
                kept_imgs.append(aug_bgr)
                print(f"[DEBUG] augment#{tries}: accepted (SIM {LOW_SIM}–{HIGH_SIM}), total accepted: {len(kept_imgs)}")
            else:
                print(f"[DEBUG] augment#{tries}: similarity {sim:.3f} out of bounds [{LOW_SIM}, {HIGH_SIM}]")


        # if none passed the filter, pick top 2
            if not kept_imgs and kept_sims:
                kept_sims.sort(key=lambda x: x[0], reverse=True)
                for sim, img in kept_sims[:FALLBACK_KEEP]:
                    kept_imgs.append(img)
                print(f"[DEBUG] Fallback accepted, sim={sim:.3f}")

        # save
        for i, img_out in enumerate(kept_imgs[:N_AUG], start=1):
            if len(glob.glob(f"{dst}/*_aug*.jpg")) >= config.MAX_AUG_PER_USER:
                break
            fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(dst, fname), img_out)
            print(f"[DEBUG] Saved: {os.path.join(dst, fname)}")

        print(f"kept {len(kept_imgs[:N_AUG])}/{N_AUG} after {tries} tries")

print("\nAugmentation complete.")
