from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtGui    import QPixmap, QImage
from PyQt5.QtCore   import Qt, pyqtSlot
import numpy as np
from ui.threads.face_capture import FaceCaptureThread
from ui.threads.voice_capture import VoiceCaptureThread
import config
from scipy.spatial.distance import cosine
import cv2


class MultiModalAuthDialog(QDialog):
    BORDER   = 4
    INNER_W  = 320
    INNER_H  = 180
    OUTER_W  = INNER_W + 2 * BORDER
    OUTER_H  = INNER_H + 2 * BORDER

    def _set_border(self, color: str):
        self.cam_view.setStyleSheet(
            f"padding:0px; border:{self.BORDER}px solid {color};")

    def _set_voice_dot(self, speaking: bool):
        colour = "lime" if (self.voice_live and speaking) else "gray"
        self.voice_dot.setStyleSheet(f"font-size:48px; color:{colour};")

    def __init__(self, parent):
        super().__init__(parent)

        self.face_svm         = parent.face_svm
        self.face_classes     = parent.face_classes
        self.global_threshold = parent.global_threshold
        self.class_thresholds = parent.class_thresholds

        self.setWindowTitle("Face + Voice Authentication")
        self.setModal(True)
        self.resize(400, 400)

        self.face_result    = None
        self.voice_result   = None
        self._last_border   = "gray"
        self._border_locked = False
        self.voice_live     = True

        main = QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(40)

        self.cam_view = QLabel(self)
        self.cam_view.setFixedSize(self.OUTER_W, self.OUTER_H)
        self.cam_view.setAlignment(Qt.AlignCenter)
        self._set_border("gray")
        main.addWidget(self.cam_view, 0, Qt.AlignCenter)

        self.face_text = QLabel("Hold still…", self)
        self.face_text.setAlignment(Qt.AlignCenter)
        main.addWidget(self.face_text)

        voice_col = QVBoxLayout()
        voice_col.setSpacing(5)

        self.voice_dot = QLabel("●", self)
        self.voice_dot.setAlignment(Qt.AlignCenter)
        self._set_voice_dot(False)
        voice_col.addWidget(self.voice_dot)

        self.voice_text = QLabel("Waiting for speech…", self)
        self.voice_text.setAlignment(Qt.AlignCenter)
        voice_col.addWidget(self.voice_text)

        main.addLayout(voice_col)

        self.face_thr = FaceCaptureThread(self, config.CAM_DEVICE,
                                          config.FRAME_SCALE)
        self.face_thr.processing_signal.connect(
            lambda: self.face_text.setText("Processing…"))
        self.face_thr.frame_signal.connect(self._update_camera)
        self.face_thr.result_signal.connect(self._on_face)
        self.face_thr.start()

        self.voice_thr = VoiceCaptureThread(self.parent(),
                                            required_speech=config.RECORD_SEC)
        self.voice_thr.speech_signal.connect(self._set_voice_dot)
        self.voice_thr.processing_signal.connect(
            lambda: self.voice_text.setText("Processing…"))
        self.voice_thr.result_signal.connect(self._on_voice_embedding)
        self.voice_thr.start()

    @pyqtSlot(str, float, object)
    def _on_face(self, name, score, probs):
        self.face_text.hide()

        print("[FaceAuth] class probabilities:")
        for cls, p in zip(self.face_classes, probs):
            print(f"    {cls}: {p:.3f}")

        thr = self.class_thresholds.get(name, self.global_threshold)
        print(f"[FaceAuth] using threshold={thr:.3f} for {name}")

        if score < thr:
            config.db.log_attempt(name, "face_stage", False)
            return self._generic_fail()

        self.face_result = (name, score)

    @pyqtSlot(np.ndarray)
    def _on_voice_embedding(self, test_emb: np.ndarray):
        if not self.face_result:
            return

        claimed_name, _ = self.face_result
        thr = self.parent().voice_thresholds.get(
            claimed_name, config.VOICE_MARGIN)
        print(f"[VoiceAuth] using voice threshold={thr:.3f} for {claimed_name}")

        genuine = config.db.get_audio_embeddings(claimed_name,
                                                 emb_dim=test_emb.size)
        if not genuine:
            return self._generic_fail()

        best_sim = max(1 - cosine(test_emb, emb) for emb in genuine)
        print(f"[VoiceAuth] best genuine={best_sim:.3f}")

        if best_sim < thr:
            config.db.log_attempt(claimed_name, "voice_stage", False)
            return self._generic_fail()

        self.voice_result = (claimed_name, best_sim)
        self._try_finish()

    def _try_finish(self):
        if self.face_result and self.voice_result:
            self.auth_name    = self.face_result[0]
            self.auth_success = True
            self._stop_threads()
            self.accept()

    def _generic_fail(self):
        self.voice_live = False
        self._set_voice_dot(False)

        mbox = QMessageBox(self)
        mbox.setIcon(QMessageBox.Warning)
        mbox.setWindowTitle("Authentication failed")
        mbox.setText(
            "<span style='font-size:14pt; font-weight:600;'>"
            "Could not verify your identity"
            "</span><br/>"
            "Please try again."
        )
        retry = mbox.addButton("Retry",  QMessageBox.AcceptRole)
        cancel = mbox.addButton("Cancel", QMessageBox.RejectRole)
        mbox.setDefaultButton(retry)
        mbox.exec_()

        if mbox.clickedButton() is retry:
            self._restart_capture()
        else:
            self.reject()
    def _restart_capture(self):
        for thr in (getattr(self, "face_thr", None),
                    getattr(self, "voice_thr", None)):
            if thr and thr.isRunning():
                thr.requestInterruption()
                thr.wait()

        self.face_text.setText("Hold still…")
        self.face_text.show()
        self.voice_live = True
        self._set_voice_dot(False)
        self.voice_text.setText("Waiting for speech…")
        self._set_border("gray")
        self._last_border   = "gray"
        self._border_locked = False

        self.face_result = self.voice_result = None

        self.face_thr = FaceCaptureThread(self, config.CAM_DEVICE,
                                          config.FRAME_SCALE)
        self.face_thr.processing_signal.connect(
            lambda: self.face_text.setText("Processing…"))
        self.face_thr.frame_signal.connect(self._update_camera)
        self.face_thr.result_signal.connect(self._on_face)
        self.face_thr.start()

        self.voice_thr = VoiceCaptureThread(self.parent(),
                                            required_speech=config.RECORD_SEC)
        self.voice_thr.speech_signal.connect(self._set_voice_dot)
        self.voice_thr.processing_signal.connect(
            lambda: self.voice_text.setText("Processing…"))
        self.voice_thr.result_signal.connect(self._on_voice_embedding)
        self.voice_thr.start()

    def reject(self):
        self._stop_threads()
        super().reject()

    def closeEvent(self, ev):
        self.reject()
        ev.accept()

    @pyqtSlot(np.ndarray, bool)
    def _update_camera(self, frame, face_found):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.cam_view.contentsRect().size(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_view.setPixmap(pix)

        if not self._border_locked:
            if face_found:
                self._set_border("lime")
                self._border_locked = True
            elif self._last_border != "gray":
                self._set_border("gray")
                self._last_border = "gray"

    def _stop_threads(self):
        for thr in (getattr(self, "face_thr", None),
                    getattr(self, "voice_thr", None)):
            if thr and thr.isRunning():
                thr.requestInterruption()
                thr.wait()
