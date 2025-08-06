import webrtcvad
import wave
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import config

def detect_speech_in_wav(path: str,
                         aggressiveness: int = 3,
                         min_segment_ms: int = 300) -> float:

    #instance of VAD engine
    vad = webrtcvad.Vad(aggressiveness)
    with wave.open(path, "rb") as wf:

        #check compatibility with webrtc
        assert wf.getnchannels() == 1  # check audio is mono
        assert wf.getsampwidth() == 2  # check each audio sample is 2 bytes
        sr = wf.getframerate()
        assert sr in (8000,16000,32000,48000) # check if the sampling rate is supported

        frame_ms   = 30
        frame_bytes= int(sr * frame_ms/1000) * 2   # how many bytes of audio are in a frame of 30 ms
        frame_s    = frame_ms/1000.0
        #in case frame_ms specified is < 30 ms, make sure its at least 1 frame
        min_frames = max(1, int(min_segment_ms/frame_ms))

        total = 0.0
        contig = 0
        while True:
            #read samples
            chunk = wf.readframes(frame_bytes//2) # each frame is 2 bytes
            #if we hit the end of file exit the loop
            if len(chunk) < frame_bytes:
                break
            if vad.is_speech(chunk, sr):
                contig += 1
            else: #in this case we reached the end of a speech segment and we check if it was long enough
                if contig >= min_frames:
                    total += contig * frame_s
                contig = 0

        #if the audio ended while the person was talking, it would never reach a no-speech so check the last part separately
        if contig >= min_frames:
            total += contig * frame_s
    return total


def record_sample(name: str, sample_idx: int, duration: float = None, fs: int = None) -> bool:

    duration = duration or config.RECORD_SEC
    fs = fs or config.VOICE_SAMPLE_RATE

    #builds the folder path
    save_dir = Path(config.RAW_VOICE_DIR) / name
    #if the directory doesnt exist, create it, also create any missing parent folders, dont crash if it is already there
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"sample_{sample_idx}.wav"

    # Record
    print(f"Recording {duration}s at {fs}Hz for '{name}' (sample {sample_idx})…")
    try:
        #total number of samples to record
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        #pause the program until recording is finished
        sd.wait()
        sf.write(str(filename), recording, fs)
    except Exception as e:
        print(f" Recording or write failed: {e}")
        return False

    #check if enough speech was detected
    speech_secs = detect_speech_in_wav(str(filename))
    if speech_secs < config.MIN_SPEECH_SECS:
        print(f"Too little speech ({speech_secs:.2f}s), deleting {filename.name}")
        try:
            #delete the file
            filename.unlink()
        except FileNotFoundError:
            #if the file isnt there, ignore
            pass
        return False

    print(f"Saved: {filename}")
    return True