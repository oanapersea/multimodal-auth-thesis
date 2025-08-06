import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import face_recognition


class FaceCaptureThread(QThread):
    result_signal     = pyqtSignal(str, float, np.ndarray)
    stable_update     = pyqtSignal(int, int)
    processing_signal = pyqtSignal()
    frame_signal      = pyqtSignal(np.ndarray, bool)
    detect_signal     = pyqtSignal(bool)

    def __init__(self, parent, cam_device, frame_scale,
                 required_stable=5, pos_tol=20, size_tol=20, poll_hz=30):
        super().__init__(parent)
        self.parent_ref       = parent
        self.cam_device       = cam_device
        self.frame_scale      = frame_scale
        self.required_stable  = required_stable
        self.pos_tol          = pos_tol
        self.size_tol         = size_tol
        self.poll_interval    = 1.0 / poll_hz

        self.cap = cv2.VideoCapture(self.cam_device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   2) #the buffer holds 2 frames the camera has already captured

        for _ in range(15):
            self.cap.read() #avoid the first laggy frames

        self._processed     = False
        self._stability     = 0
        self._last_center   = None
        self._last_size     = None

    def run(self):
        while not self.isInterruptionRequested():
            ok, frame = self.cap.read()
            if not ok:
                continue

            small = cv2.resize(
                frame, (0, 0), #automatically calculate new size from scaling features
                fx=self.frame_scale, fy=self.frame_scale,
                interpolation=cv2.INTER_AREA #downsampling method
            )
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            face_found = bool(locs)

            self.frame_signal.emit(frame, face_found) #signal to authentication page to update preview
            self.detect_signal.emit(face_found)

            if not face_found:
                self._stability = 0 #restart stability counter
                self._last_center = self._last_size = None #resets the memory of the last face's position and size
                time.sleep(self.poll_interval)
                continue

            top, right, bottom, left = locs[0]
            w, h = right - left, bottom - top
            cx, cy = left + w / 2, top + h / 2  # computes the center pointof the face

            if self._last_center is None:  # if its the first face seen, set stability to 1
                self._last_center = (cx, cy)
                self._last_size = (w, h)
                self._stability = 1
            else:
                dx = abs(cx - self._last_center[0]) # measure how much the face center has moved since the last frame
                dy = abs(cy - self._last_center[1])
                dw = abs(w - self._last_size[0]) # measure how much the face size has changes
                dh = abs(h - self._last_size[1])
                # face is still stable
                if dx < self.pos_tol and dy < self.pos_tol \
                        and dw < self.size_tol and dh < self.size_tol:
                    self._stability += 1
                else:
                    self._last_center = (cx, cy) #face has moved too much
                    self._last_size = (w, h)
                    self._stability = 1

            self.stable_update.emit(self._stability, self.required_stable)

            if self._stability >= self.required_stable and not self._processed:
                self.processing_signal.emit() #emit to the authentication page to print "Processing"

                emb = face_recognition.face_encodings(rgb, [locs[0]])[0]
                probs = self.parent_ref.face_svm.predict_proba([emb])[0]
                idx = int(np.argmax(probs)) #max of the probabilities returned by the svm
                name = self.parent_ref.face_classes[idx]
                score = float(probs[idx])

                self.result_signal.emit(name, score, probs) #emit to the authentication the user and probability
                self._processed = True

            time.sleep(self.poll_interval)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
