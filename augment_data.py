import sys
import warnings
import numpy as np
import librosa
import soundfile as sf
from resemblyzer import preprocess_wav
from config import encoder
import config
import tempfile
SR        = config.VOICE_SAMPLE_RATE
RAW_DIR   = config.CLEAN_VOICE_DIR
AUG_DIR   = config.AUG_VOICE_DIR
N_AUG     = config.N_AUG
MAX_TRIES = config.MAX_TRIES
LOW_SIM   = config.LOW_SIM
HIGH_SIM  = config.HIGH_SIM

#process one speaker or all speakers
speaker = sys.argv[1] if len(sys.argv) > 1 else None

#compare speaker embeddings
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_augmented(y_aug: np.ndarray, sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, y_aug, sr)
        wav_proc = preprocess_wav(tmp.name)
        return encoder.embed_utterance(wav_proc)


def augment_clip(y: np.ndarray, sr: int) -> np.ndarray:
    choice = np.random.choice(["stretch", "pitch", "noise"])
    if choice == "pitch":
        n_steps = np.random.uniform(-0.5, 0.5)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    elif choice == "stretch":
        if np.random.rand() < 0.8:
            rate = np.random.uniform(0.9, 1.1)
        else:
            rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y, rate=rate)
    else:
        snr_db = np.random.uniform(17, 20)
        rms = np.sqrt(np.mean(y**2))
        noise = np.random.randn(len(y)) * rms * 10**(-snr_db/20)
        return y + noise

def batch_augment(speaker: str = None):
    #either go through all files or only a user's files
    if speaker:
        base = RAW_DIR / speaker
    else:
        base = RAW_DIR

    for wav_path in sorted(base.rglob("*.wav")):
        spk     = wav_path.parent.name
        out_dir = AUG_DIR / spk

        #create the directory if it dosnt exist
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            #load the audio file into a NumPy array
            y, _    = librosa.load(str(wav_path), sr=SR)
            #compute speaker embedding
            emb_o   = encoder.embed_utterance(preprocess_wav(str(wav_path)))
        except Exception as e:
            warnings.warn(f"⚠️ Failed to load/embed {wav_path.name}: {e}")
            continue

        accepted = 0
        tries    = 0
        #we have a maximum nr of tries to get a certain number of audio_augmented clips
        while accepted < N_AUG and tries < MAX_TRIES:
            y_aug = augment_clip(y, SR)
            try:
                emb_a = embed_augmented(y_aug, SR)
            except Exception as e:
                warnings.warn(f"⚠️ Embed failed on augment of {wav_path.name}: {e}")
                tries += 1
                continue

            #check similarity between original and audio_augmented, if it is too low do not save it, to not confuse the model
            sim = cos_sim(emb_o, emb_a)
            if LOW_SIM <= sim <= HIGH_SIM:
                if len(list(out_dir.glob("*_aug*.wav"))) >= config.MAX_AUG_PER_USER:
                    break
                fname = f"{wav_path.stem}_aug{accepted+1}.wav"
                out_path = out_dir / fname
                sf.write(str(out_path), y_aug, SR)
                print(f" Kept {fname} (sim={sim:.3f})")
                accepted += 1
            else:
                print(f"Rejected {wav_path.name} sim={sim:.3f}")
            tries += 1

        if accepted < N_AUG:
            warnings.warn(f" Only {accepted}/{N_AUG} augments passed for {wav_path.name}")


if __name__ == "__main__":
    batch_augment(speaker)
