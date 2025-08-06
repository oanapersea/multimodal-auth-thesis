import sys
import sqlite3
import db
from ui.widgets.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import qdarkstyle

db.init_db()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())