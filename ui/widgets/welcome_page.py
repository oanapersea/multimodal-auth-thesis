from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

class WelcomePage(QWidget):
    def __init__(self, on_logout, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.wel_lbl = QLabel("")
        self.wel_lbl.setStyleSheet("font-size:28px;font-weight:bold;")
        layout.addWidget(self.wel_lbl)

        logout_btn = QPushButton("Logout")
        logout_btn.setFixedSize(160, 50)
        logout_btn.clicked.connect(on_logout)
        layout.addWidget(logout_btn, alignment=Qt.AlignCenter)
