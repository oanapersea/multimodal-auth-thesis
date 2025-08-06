from PyQt5.QtCore import QThread, pyqtSignal
from record import record_sample


class RecorderThread(QThread):

    result = pyqtSignal(bool)

    def __init__(self, username: str, sample_idx: int, duration: float):
        super().__init__()
        self.username   = username
        self.sample_idx = sample_idx
        self.duration   = duration

    def run(self):
        ok = record_sample(self.username, self.sample_idx, duration=self.duration)
        self.result.emit(ok)