from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore    import Qt

class LoginPage(QWidget):
    def __init__(self, on_auth, on_enroll, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(20)

        title = QLabel("Welcome!")
        title.setStyleSheet("font-size:32px;font-weight:bold;")
        layout.addWidget(title, alignment=Qt.AlignCenter)
        layout.addSpacing(30)

        row = QHBoxLayout()
        row.setSpacing(30)

        auth_btn = QPushButton("Authenticate")
        auth_btn.setFixedSize(200, 50)
        auth_btn.setCursor(Qt.PointingHandCursor)
        auth_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                font-size: 16px;
                font-weight: normal;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2C5F8E;
            }
        """)
        auth_btn.clicked.connect(on_auth)
        row.addWidget(auth_btn, alignment=Qt.AlignCenter)
        layout.addLayout(row)

        enroll_btn = QPushButton("I am not enrolled")
        enroll_btn.setCursor(Qt.PointingHandCursor)
        enroll_btn.setFlat(True)
        enroll_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #007BFF;
                text-decoration: underline;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #0056AA;
            }
            QPushButton:pressed {
                color: #004488;
            }
        """)
        enroll_btn.clicked.connect(on_enroll)
        layout.addSpacing(20)
        layout.addWidget(enroll_btn, alignment=Qt.AlignCenter)
