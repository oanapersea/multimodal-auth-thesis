import os
import sys
from glob import glob
import noisereduce as nr
import soundfile as sf
import librosa
import numpy as np
import config
RAW_DIR   = config.RAW_VOICE_DIR
CLEAN_DIR = config.CLEAN_VOICE_DIR
SR        = config.VOICE_SAMPLE_RATE

#we can process one speaker or all
speaker = sys.argv[1] if len(sys.argv) > 1 else None

def denoise_file(in_path, out_path):
    #audio data, sample rate
    y, sr = sf.read(in_path)
    #resample if the sampling rate is not 16000
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR

    #convert stereo to mono if necessary
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    #use the first 0.5 seconds of the audio as noise
    noise_clip = y[: int(0.5 * sr)]

    #denoise audio
    reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)

    #create directories if they dont exist already
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    #write the file
    sf.write(out_path, reduced, sr)
    print(f"Denoised: {out_path}")


def batch_denoise(speaker=None):
    #either go through all files or only a user's files
    if speaker:
        pattern = os.path.join(RAW_DIR, speaker, "*.wav")
    else:
        pattern = os.path.join(RAW_DIR, "*", "*.wav")
    #find all matching files
    files = glob(pattern)
    print(f"Found {len(files)} files to process")

    for in_path in files:
        speaker_name = os.path.basename(os.path.dirname(in_path))
        fname        = os.path.basename(in_path)
        out_dir      = os.path.join(CLEAN_DIR, speaker_name)
        out_path     = os.path.join(out_dir, fname)


        if os.path.exists(out_path):
            print(f" Skipping (already done): {out_path}")
            continue

        denoise_file(in_path, out_path)


if __name__ == "__main__":
    batch_denoise(speaker)
