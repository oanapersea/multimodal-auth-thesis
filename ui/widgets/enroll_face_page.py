from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

class EnrollFacePage(QWidget):
    def __init__(self, on_snapshot, on_back, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.setSpacing(10)

        h2 = QLabel("Face Enrollment")
        h2.setStyleSheet("font-size:28px;font-weight:bold;")
        h2.setAlignment(Qt.AlignCenter)
        layout.addWidget(h2)
        layout.setSpacing(10)

        self.face_preview = QLabel()
        self.face_preview.setFixedSize(320, 240)
        layout.addWidget(self.face_preview)

        self.snap_lbl = QLabel("Snapshot 1 of 5")
        self.snap_lbl.setStyleSheet("font-size:18px;")
        layout.addWidget(self.snap_lbl)

        self.snap_btn = QPushButton("Capture Snapshot")
        self.snap_btn.setFixedSize(240, 60)
        self.snap_btn.setStyleSheet("""
           QPushButton { background-color: rgba(55, 255, 55, 0.7); color: white; font-size:18px; border: none; border-radius: 6px;}
           QPushButton:hover { background-color: rgba(55, 255, 55, 0.8);}
           QPushButton:pressed { background-color: rgba(55, 255, 55, 0.9);}
        """)
        self.snap_btn.clicked.connect(on_snapshot)
        layout.addWidget(self.snap_btn)
        layout.addSpacing(20)

        back_btn = QPushButton("← Restart enrollment")
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
