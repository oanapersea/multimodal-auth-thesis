import db
import joblib
from resemblyzer import VoiceEncoder
from pathlib import Path

for bak in Path("models").glob("*.joblib.bak"):
    try:
        bak.unlink()
    except Exception as e:
        print(f"Could not remove {bak}: {e}")

BASE_DIR       = Path(__file__).parent
RAW_FACE_DIR   = BASE_DIR / "data/images/images_raw"
PROC_FACE_DIR  = BASE_DIR / "data/images/images_processed"
AUG_FACE_DIR  = BASE_DIR / "data/images/images_augmented"

RAW_VOICE_DIR  = BASE_DIR / "data/audio/audio_raw"
CLEAN_VOICE_DIR      = BASE_DIR / "data/audio/audio_cleaned"
AUG_VOICE_DIR        = BASE_DIR / "data/audio/audio_augmented"

MODELS_DIR     = BASE_DIR / "models"
VOICE_MODEL_FILE    = MODELS_DIR / "voice_thresholds.joblib"
FACE_MODEL_FILE     = MODELS_DIR / "face_svm.joblib"
DB_PATH = str(BASE_DIR / "auth.db")

MAX_AUG_PER_USER = 25
MIN_SPEECH_SECS= 3.0
LOW_SIM        = 0.85
HIGH_SIM       = 0.99
N_AUG          = 5
MAX_TRIES      = 15

OUTPUT_SIZE    = (160,160)
MARGIN_FRAC    = 0.2
DETECTION_MODEL= "hog"
CAM_DEVICE     = 0
FRAME_SCALE    = 0.25

RECORD_SEC     = 5
VOICE_SAMPLE_RATE = 16000
VOICE_MARGIN = 0.20

encoder = VoiceEncoder()


