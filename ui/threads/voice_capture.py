from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import sounddevice as sd
import webrtcvad
import config

class VoiceCaptureThread(QThread):
    speech_signal     = pyqtSignal(bool)
    processing_signal = pyqtSignal()
    result_signal = pyqtSignal(np.ndarray)
    no_voice          = pyqtSignal()

    def __init__(self, parent, fs=config.VOICE_SAMPLE_RATE, aggressiveness=2, required_speech=config.RECORD_SEC):
        super().__init__(parent)
        self.fs              = fs
        self.vad             = webrtcvad.Vad(aggressiveness)
        self.required_speech = required_speech
        self.block_ms        = 30
        self.block_size      = int(self.fs * self.block_ms / 1000)
        self.total_speech    = 0.0
        self.buffer          = []

    def run(self):
        try:
            stream = sd.RawInputStream(
                samplerate=self.fs,
                blocksize=self.block_size,
                dtype='int16',
                channels=1
            )
            stream.start()
        except Exception as e:
            print(" Audio open error:", e)
            self.no_voice.emit()
            return

        while not self.isInterruptionRequested() and self.total_speech < self.required_speech:
            try:
                data, _ = stream.read(self.block_size)
            except Exception as e:
                print("️Audio read error:", e)
                break

            is_sp = self.vad.is_speech(data, self.fs)
            if is_sp:
                self.total_speech += (self.block_ms / 1000.0)
            self.speech_signal.emit(is_sp) #emit to the authentication page that speech is detected -> turn the dot green

            pcm = np.frombuffer(data, dtype=np.int16) #convert to numpy
            block = (pcm.astype(np.float32) / 32768.0).reshape(-1,1) #normalizez the audio
            self.buffer.append(block) #build full sample

        stream.stop()
        stream.close()

        if self.isInterruptionRequested():
            return

        if self.total_speech < self.required_speech:
            self.no_voice.emit()
            return

        self.processing_signal.emit() #emit to the authentication page to print processing

        audio = np.concatenate(self.buffer, axis=0).flatten()

        try:
            emb = config.encoder.embed_utterance(audio.astype(np.float32))
            self.result_signal.emit(emb) #emit to the authentication page the voice embedding
        except Exception:
            self.no_voice.emit()
