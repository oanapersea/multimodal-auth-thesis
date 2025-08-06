from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt

class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Enrollment…")
        self.setModal(True)
        v = QVBoxLayout(self)
        lbl = QLabel("Please wait...", self)
        lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(lbl)
        bar = QProgressBar(self)
        bar.setRange(0,0)
        v.addWidget(bar)
        self.setFixedSize(300,100)