import sys
import subprocess
import joblib
import face_recognition
from resemblyzer import preprocess_wav
from PyQt5.QtCore import QThread, pyqtSignal
import config
import shutil
from pathlib import Path
from config import VOICE_MODEL_FILE

def _purge_user_folders(username: str):
    for root in (
        config.RAW_VOICE_DIR,
        config.CLEAN_VOICE_DIR,
        config.AUG_VOICE_DIR,
        config.RAW_FACE_DIR,
        config.PROC_FACE_DIR,
        config.AUG_FACE_DIR,
    ):
        p = Path(root) / username
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"removed {p}")
        except Exception as e:
            print(f"Could not remove {p}: {e}")


class EnrollmentPipelineThread(QThread):
    result = pyqtSignal(bool)

    MODEL_FILES = [
        config.FACE_MODEL_FILE,
        config.VOICE_MODEL_FILE,
    ]

    def __init__(self, username, parent=None):
        super().__init__(parent)
        if not username:
            raise ValueError("EnrollmentPipelineThread got an empty username")
        self.username = username
        self.BASE_DIR = config.BASE_DIR
        self.AUG_FACE_DIR = config.AUG_FACE_DIR
        self.AUG_VOICE_DIR = config.AUG_VOICE_DIR
        self.RAW_VOICE_DIR = config.RAW_VOICE_DIR
        self.RAW_FACE_DIR = config.RAW_FACE_DIR
        self.PROC_FACE_DIR = config.PROC_FACE_DIR
        self.CLEAN_VOICE_DIR = config.CLEAN_VOICE_DIR
        self.db = config.db
        self.encoder = config.encoder

    def run(self):
        u = self.username
        self._make_backups()
        try:
            self._pipeline(u)
            self._final_train()
            self._delete_backups()
            self.result.emit(True)
        except Exception as e:
            print("[Enroll] ERROR:", e)
            self._rollback(u)
            self._restore_backups()
            self.result.emit(False)

    def _pipeline(self, u):
        subprocess.run(
            [sys.executable, str(self.BASE_DIR / "denoise_audio.py"), u], check=True
        )
        subprocess.run(
            [sys.executable, str(self.BASE_DIR / "augment_data.py"), u], check=True
        )

        cleaned_dir = self.CLEAN_VOICE_DIR / u
        for wav_path in cleaned_dir.glob("*.wav"):
            orig = wav_path.stem
            wav = preprocess_wav(str(wav_path))
            emb = self.encoder.embed_utterance(wav)
            self.db.add_audio_embedding(u, emb.tobytes(), orig_id=orig, is_augmented=0)

        aug_dir = self.AUG_VOICE_DIR / u
        if aug_dir.exists():
            for wav_path in aug_dir.glob("*.wav"):
                orig = wav_path.stem.split("_aug")[0]
                wav = preprocess_wav(str(wav_path))
                emb = self.encoder.embed_utterance(wav)
                self.db.add_audio_embedding(
                    u, emb.tobytes(), orig_id=orig, is_augmented=1
                )
        else:
            print(f"No augmented audio for {u}")

        subprocess.run(
            [sys.executable, str(self.BASE_DIR / "preprocess_faces.py"), u], check=True
        )

        face_dir = self.PROC_FACE_DIR / u
        if not face_dir.exists():
            print(f" No processed faces for {u}")
            return

        subprocess.run(
            [sys.executable, str(self.BASE_DIR / "augment_faces.py"), u], check=True
        )

        for img_path in face_dir.glob("*.jpg"):
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                stem = img_path.stem
                self.db.add_face_embedding(
                    u, encs[0].tobytes(), orig_id=stem, is_augmented=0
                )
                print(f"[Face ] embedding {img_path.name}")
            else:
                print(f"No images_raw in {img_path.name}")

        aug_dir = config.AUG_FACE_DIR/u
        if aug_dir.exists():
            for img_path in aug_dir.glob("*.jpg"):
                img = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img)
                if encs:
                    stem = img_path.stem.split("_aug")[0]
                    self.db.add_face_embedding(
                        u, encs[0].tobytes(), orig_id=stem, is_augmented=1
                    )
                    print(f"[Face] embedding AUG {img_path.name}")
        else:
            print(f"No augmented faces for {u}")

    def _rollback(self, u):
        try:
            self.db.delete_user_data(u)
            print(f"[Enroll] Rolled back DB rows for {u}")
        except Exception as e:
            print("[Enroll] DB rollback failed:", e)

        _purge_user_folders(u)
        self._delete_backups()

    def _make_backups(self):
        for f in self.MODEL_FILES:
            if f.exists():
                shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))

    def _restore_backups(self):
        for f in self.MODEL_FILES:
            bak = f.with_suffix(f.suffix + ".bak")
            if bak.exists():
                shutil.move(str(bak), str(f))

    def _delete_backups(self):
        for f in self.MODEL_FILES:
            bak = f.with_suffix(f.suffix + ".bak")
            if bak.exists():
                bak.unlink()

    def _final_train(self):
        subprocess.run(
            [sys.executable, str(self.BASE_DIR / "train_classifier_svm.py")], check=True
        )
        subprocess.run(
            [sys.executable, str(self.BASE_DIR / "compute_voice_thresholds.py")],
            check=True,
        )
        vd = joblib.load(VOICE_MODEL_FILE)
        self.parent().voice_thresholds = vd.get("voice_thresholds", {})
        _purge_user_folders(self.username)

