from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt

class EnrollAudioPage(QWidget):
    def __init__(self, on_record, on_back, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.setContentsMargins(0, 60, 0, 0)

        h1 = QLabel("Audio Enrollment")
        h1.setAlignment(Qt.AlignCenter)
        h1.setStyleSheet("font-size:28px;font-weight:bold;")
        layout.addWidget(h1)
        layout.addSpacing(40)

        self.en_name_input = QLineEdit()
        self.en_name_input.setPlaceholderText("Enter new username")
        self.en_name_input.setFixedWidth(300)
        layout.addWidget(self.en_name_input, alignment=Qt.AlignCenter)
        layout.addSpacing(30)

        self.sample_lbl = QLabel("Sample 1 of 5")
        self.sample_lbl.setStyleSheet("font-size:18px;")
        layout.addWidget(self.sample_lbl)

        self.rec_btn = QPushButton("● Record Sample")
        self.rec_btn.setFixedSize(220, 60)
        self.rec_btn.setStyleSheet("font-size:18px; background:#d9534f; color:white;")
        self.rec_btn.clicked.connect(on_record)
        layout.addWidget(self.rec_btn)
        layout.addSpacing(40)

        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(on_back)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 122, 255, 0.6);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: rgba(0, 122, 255, 0.8);}
            QPushButton:pressed { background-color: rgba(0, 122, 255, 0.8);}
        """)
        layout.addWidget(back_btn, alignment=Qt.AlignLeft)
