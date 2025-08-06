import shutil
import joblib
import cv2
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import face_recognition
import config
from config import (
    VOICE_MODEL_FILE, FACE_MODEL_FILE, RAW_VOICE_DIR,
    CLEAN_VOICE_DIR, AUG_VOICE_DIR, RAW_FACE_DIR,  PROC_FACE_DIR, AUG_FACE_DIR, RECORD_SEC, FRAME_SCALE, db,
    CAM_DEVICE
)
from ui.threads.enrollment import EnrollmentPipelineThread
from ui.threads.recorder import RecorderThread
from ui.dialogs.processing import ProcessingDialog
from ui.dialogs.authentication import MultiModalAuthDialog
from .login_page import LoginPage
from .enroll_audio_page import EnrollAudioPage
from .enroll_face_page import EnrollFacePage
from .welcome_page import WelcomePage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biometric Login")
        self.resize(600, 500)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_page = LoginPage(self.on_multimodal_auth, self.show_enroll_audio_page)
        self.audio_page = EnrollAudioPage(self._record_sample, self.cancel_enroll)
        self.face_page = EnrollFacePage(self._capture_snapshot, self.cancel_enroll)
        self.welcome_page = WelcomePage(self.show_login_page)

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.audio_page)
        self.stack.addWidget(self.face_page)
        self.stack.addWidget(self.welcome_page)

        #"Sample 1 of 5"
        self._audio_count = 1
        #"Snapshot 1 of 5"
        self._face_count = 1
        self._pending_username = None

        self.load_models()
        self.show_login_page()

    #the models are loaded only once, when the app starts
    #each dialog or thread that needs the model can grab it directly
    def load_models(self):
        vd = joblib.load(VOICE_MODEL_FILE)
        self.voice_clf = vd["svm"]
        self.voice_classes = vd["classes"]
        self.voice_thresholds = vd.get("voice_thresholds", {})

        fd = joblib.load(FACE_MODEL_FILE)
        self.face_svm = fd["svm"]
        self.face_classes = fd["classes"]
        self.global_threshold = fd.get("global_threshold", fd.get("threshold"))
        self.class_thresholds = fd.get("class_thresholds", {})

    def closeEvent(self, event):
        if getattr(self, "_pending_username", None):
            self.cancel_enroll()
        super().closeEvent(event) #built-in closeEvent that destroys the windows, signals etc


    def show_login_page(self):
        self.stack.setCurrentIndex(0)

    def show_enroll_audio_page(self):
        self._pending_username = None
        self._audio_count = 1
        self.audio_page.en_name_input.clear()
        self.audio_page.en_name_input.setEnabled(True)
        self.audio_page.sample_lbl.setText("Sample 1 of 5")
        self.audio_page.rec_btn.setEnabled(True)
        self.audio_page.rec_btn.setText("● Record Sample")
        self.audio_page.rec_btn.setStyleSheet("font-size:18px; background:#d9534f; color:white;")
        self.stack.setCurrentIndex(1)

    def show_enroll_face_page(self):
        self._face_count = 1
        self.face_page.snap_lbl.setText("Snapshot 1 of 5")
        self.face_page.snap_btn.setEnabled(True)
        self.cap = cv2.VideoCapture(CAM_DEVICE)
        self.timer = QTimer(self) #creates a timer object and it stops once the window stops (self)
        self.timer.timeout.connect(self._update_preview)
        self.timer.start(30) #smooth preview of the camera update
        self.stack.setCurrentIndex(2)

    def show_welcome_page(self, name):
        self.welcome_page.wel_lbl.setText(f"Welcome, {name}!")
        self.stack.setCurrentIndex(3)

    def cancel_enroll(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        u = getattr(self, "_pending_username", None)
        if u:
            db.delete_user_data(u)
            for p in [
                RAW_VOICE_DIR / u,
                CLEAN_VOICE_DIR / u,
                AUG_VOICE_DIR / u,
                RAW_FACE_DIR / u,
                PROC_FACE_DIR / u,
                AUG_FACE_DIR / u
            ]:
                if p.exists() and p.is_dir() and p.name == u:
                    shutil.rmtree(p)
        self._pending_username = None
        self.show_login_page()

    def _record_sample(self):
        if self._audio_count == 1:
            name = self.audio_page.en_name_input.text().strip().lower()
            if not name:
                QMessageBox.warning(self, "Error", "Please enter a username first.")
                return
            if db.user_exists(name):
                QMessageBox.warning(self, "Oops", f"User '{name}' already exists.")
                return
            self._pending_username = name
            (RAW_VOICE_DIR / name).mkdir(parents=True, exist_ok=True)
            self.audio_page.en_name_input.setEnabled(False)

        self.audio_page.rec_btn.setEnabled(False)
        self.audio_page.rec_btn.setText("Recording…")
        self.audio_page.rec_btn.setStyleSheet(
            "font-size:18px; background: #f0ad4e; color: white;"
        )

        self._rec_thread = RecorderThread(self._pending_username, self._audio_count, RECORD_SEC)
        self._rec_thread.result.connect(self._on_record_result)
        self._rec_thread.start()

    def _on_record_result(self, ok: bool):
        self.audio_page.rec_btn.setText("● Record Sample")
        self.audio_page.rec_btn.setStyleSheet("font-size:18px; background:#d9534f; color:white;")
        if not ok:
            QMessageBox.warning(self, "No speech detected", "We didn’t hear enough speech. Please try again.")
            self.audio_page.rec_btn.setEnabled(True)
            return

        self._audio_count += 1
        if self._audio_count <= 5:
            self.audio_page.sample_lbl.setText(f"Sample {self._audio_count} of 5")
            self.audio_page.rec_btn.setEnabled(True)
        else:
            self.show_enroll_face_page()

    def _update_preview(self):
        ret, frame = self.cap.read()
        if not ret: #failed to grab a frame
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape #ch=3 (R,G,B)
        bytes_per_line = ch * w #each line has w pixels, each pixel has ch channels (3 bytes per pixel)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(320, 240, Qt.KeepAspectRatio)
        self.face_page.face_preview.setPixmap(pix)

    def _capture_snapshot(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Error", "Failed to grab frame")
            return
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        if not locs:
            QMessageBox.warning(self, "No face detected", "Please position your face fully in view.")
            return
        username = self._pending_username
        out_dir = config.RAW_FACE_DIR / username
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir/ f"img_{self._face_count}.jpg"
        cv2.imwrite(str(path), frame)
        self._face_count += 1

        if self._face_count <= 5:
            self.face_page.snap_lbl.setText(f"Snapshot {self._face_count} of 5")
        else:
            self.timer.stop()
            self.cap.release()
            proc_dlg = ProcessingDialog(self)
            thread = EnrollmentPipelineThread(self._pending_username, parent=self)
            thread.result.connect(lambda ok, dlg=proc_dlg: self._on_pipeline_finished(ok, dlg))
            thread.start()
            proc_dlg.exec_()

    def _on_pipeline_finished(self, ok: bool, proc_dlg):
        proc_dlg.accept()
        if not ok:
            QMessageBox.critical(
                self, "Enrollment failed",
                "Something went wrong. All partial data was rolled back."
            )
            self.cancel_enroll()
            return

        enrolled_user = self._pending_username
        self._pending_username = None
        self.load_models()
        QMessageBox.information(
            self, "Enrollment complete",
            f"User '{enrolled_user}' has been enrolled!"
        )
        self.show_login_page()

    def on_multimodal_auth(self):
        dlg = MultiModalAuthDialog(self)
        success = dlg.exec_() == dlg.Accepted and dlg.auth_success
        if success:
            config.db.log_attempt(dlg.auth_name, "multimodal", True)
            self.show_welcome_page(dlg.auth_name)


