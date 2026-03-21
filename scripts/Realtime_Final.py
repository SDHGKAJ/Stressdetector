import cv2
import numpy as np
import tensorflow as tf
import joblib
import time
import os
import sys
import threading
import pandas as pd
from collections import deque

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from modules.face_eye_detector import FaceEyeDetector

cog_model = joblib.load("models/cogload_model_lgb_gpu.joblib")
cog_scaler = joblib.load("models/cogload_scaler.joblib")
eye_model = tf.keras.models.load_model("models/eye_state_model.keras")
yawn_model = tf.keras.models.load_model("models/yawn_model.keras")
stress_model = tf.keras.models.load_model("models/stress_model.keras")

IMG_SIZE = 224
WINDOW = 15
EYE_CLOSED_TH = 0.4
YAWN_TH = 0.65
DROWSY_TH = 0.35
FEATURE_NAMES = [
    "Pupil_Dilation",
    "Blink_Rate",
    "Fixation_Duration",
    "Saccade_Duration"
]

LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

cog_buffer = deque(maxlen=15)
stress_buffer = deque(maxlen=WINDOW)
eye_buffer = deque(maxlen=10)
yawn_buffer = deque(maxlen=10)
blink_history = deque(maxlen=20)
saccade_durations = deque(maxlen=10)

import mediapipe as mp
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def get_eye_box(lm, indices, frame_w, frame_h, pad=10):
    pts = np.array([(lm[i].x * frame_w, lm[i].y * frame_h) for i in indices])
    x1 = max(0, int(pts[:,0].min()) - pad)
    y1 = max(0, int(pts[:,1].min()) - pad)
    x2 = min(frame_w, int(pts[:,0].max()) + pad)
    y2 = min(frame_h, int(pts[:,1].max()) + pad)
    return x1, y1, x2 - x1, y2 - y1

def cog_label(score):
    if score < 0.85:
        return "LOW", (0,0,255)
    elif score < 1.05:
        return "MEDIUM", (0,255,255)
    return "HIGH", (0,255,0)

def stress_label(score):
    if score < 0.35:
        return "LOW", (0,255,0)
    elif score < 0.65:
        return "MEDIUM", (0,255,255)
    return "HIGH", (0,0,255)

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

class InferenceEngine:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.cog_score = -0.2
        self.stress_score = 0.0
        self.micro_score = 0.0
        self.mouth_rect = None
        self.eye_rects = []
        self.running = True
        self.fixation_start = None
        self.last_gaze = None
        self.detector = FaceEyeDetector()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_scores(self):
        with self.result_lock:
            return self.cog_score, self.stress_score, self.micro_score

    def get_mouth_rect(self):
        with self.result_lock:
            return self.mouth_rect

    def get_eye_rects(self):
        with self.result_lock:
            return list(self.eye_rects)

    def _run(self):
        while self.running:
            with self.lock:
                frame = self.frame
            if frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(rgb)

            cog_score = self.cog_score
            saccade_duration = 0
            new_eye_rects = []

            if mesh_results.multi_face_landmarks:
                lm = mesh_results.multi_face_landmarks[0].landmark
                left_eye_idx = [33,160,158,133,153,144]
                left_eye = np.array([(lm[i].x, lm[i].y) for i in left_eye_idx])
                ear = eye_aspect_ratio(left_eye)
                blink = int(ear < 0.2)
                blink_history.append(blink)
                pupil_dilation = np.linalg.norm(left_eye[1] - left_eye[4])
                blink_rate = sum(blink_history) / len(blink_history)
                gaze = np.array([lm[468].x, lm[468].y])
                fixation_duration = 0
                if self.last_gaze is not None:
                    movement = np.linalg.norm(gaze - self.last_gaze)
                    if movement < 0.002:
                        self.fixation_start = self.fixation_start or time.time()
                        fixation_duration = time.time() - self.fixation_start
                        saccade_duration = np.mean(saccade_durations) if saccade_durations else 0
                    else:
                        if self.fixation_start:
                            saccade_durations.append(time.time() - self.fixation_start)
                        fixation_duration = min(time.time() - self.fixation_start, 2.0) if self.fixation_start else 0
                        self.fixation_start = None
                        saccade_duration = np.mean(saccade_durations) if saccade_durations else 0
                self.last_gaze = gaze
                features = pd.DataFrame([{
                    "Pupil_Dilation": pupil_dilation,
                    "Blink_Rate": blink_rate,
                    "Fixation_Duration": fixation_duration,
                    "Saccade_Duration": saccade_duration
                }])
                scaled = cog_scaler.transform(features)
                scaled_df = pd.DataFrame(scaled, columns=FEATURE_NAMES)
                cog_buffer.append(cog_model.predict(scaled_df)[0])
                cog_score = np.mean(cog_buffer)

                # --- Eye boxes from MediaPipe landmarks ---
                eye_closed_votes = []
                for eye_indices in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                    ex, ey, ew, eh = get_eye_box(lm, eye_indices, w, h, pad=8)
                    if ew > 0 and eh > 0:
                        eye_roi = frame[ey:ey+eh, ex:ex+ew]
                        if eye_roi.size > 0:
                            eye_img = cv2.resize(eye_roi, (128,128)) / 255.0
                            eye_img = np.expand_dims(eye_img.astype(np.float32), 0)
                            prob = eye_model.predict(eye_img, verbose=0)[0][0]
                            is_closed = bool(prob > EYE_CLOSED_TH)
                            eye_closed_votes.append(is_closed)
                            new_eye_rects.append((ex, ey, ew, eh, is_closed))
                if eye_closed_votes:
                    eye_buffer.append(int(np.mean(eye_closed_votes) > 0.5))
                else:
                    eye_buffer.append(0)

            stress_score = self.stress_score
            micro_score = self.micro_score
            face_dets = self.detector.detect(frame)
            if face_dets:
                fx, fy, fw, fh = face_dets[0]["face"]
                face = frame[fy:fy+fh, fx:fx+fw]
                if face.size > 0:
                    face_img = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
                    face_img = np.expand_dims(face_img.astype(np.float32), 0)
                    stress_prob = stress_model.predict(face_img, verbose=0)[0][0]
                    stress_buffer.append(stress_prob)
                    stress_score = np.mean(stress_buffer)

                    fh_local, fw_local = face.shape[:2]
                    mouth_y1 = int(fh_local * 0.62)
                    mouth_y2 = int(fh_local * 0.92)
                    mouth_x1 = int(fw_local * 0.25)
                    mouth_x2 = int(fw_local * 0.75)
                    mouth_roi = face[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                    if mouth_roi.size > 0:
                        yawn_img = cv2.resize(mouth_roi, (128,128)) / 255.0
                        yawn_img = np.expand_dims(yawn_img.astype(np.float32), 0)
                        yawn = bool(yawn_model.predict(yawn_img, verbose=0)[0][0] > YAWN_TH)
                        yawn_buffer.append(int(yawn))
                        frame_mx1 = fx + mouth_x1
                        frame_my1 = fy + mouth_y1
                        frame_mx2 = fx + mouth_x2
                        frame_my2 = fy + mouth_y2
                        with self.result_lock:
                            self.mouth_rect = (frame_mx1, frame_my1, frame_mx2, frame_my2, yawn)

                eye_ratio = sum(eye_buffer) / max(len(eye_buffer), 1)
                yawn_ratio = sum(yawn_buffer) / max(len(yawn_buffer), 1)
                micro_score = round((0.4 * eye_ratio) + (0.6 * yawn_ratio), 2)

            with self.result_lock:
                self.cog_score = cog_score
                self.stress_score = stress_score
                self.micro_score = micro_score
                self.eye_rects = new_eye_rects

    def stop(self):
        self.running = False

stream = CameraStream(0)
engine = InferenceEngine()

while True:
    ret, frame = stream.read()
    if not ret:
        break

    engine.submit(frame)
    cog_score, stress_score, micro_score = engine.get_scores()

    mouth_rect = engine.get_mouth_rect()
    if mouth_rect:
        mx1, my1, mx2, my2, is_yawn = mouth_rect
        mouth_color = (0, 0, 255) if is_yawn else (0, 255, 0)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), mouth_color, 2)

    eye_rects = engine.get_eye_rects()
    for (ex, ey, ew, eh, is_closed) in eye_rects:
        eye_color = (0, 0, 255) if is_closed else (0, 255, 0)
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), eye_color, 2)

    cog_text, cog_color = cog_label(cog_score)
    stress_text, stress_color = stress_label(stress_score)
    drowsy_text = "DROWSY" if micro_score > DROWSY_TH else "ALERT"
    drowsy_color = (0,0,255) if micro_score > DROWSY_TH else (0,255,0)

    cv2.putText(frame, f"CogLoad: {cog_text} ({cog_score:.2f})", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cog_color, 2)
    cv2.putText(frame, f"Drowsy: {drowsy_text} ({micro_score:.2f})", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, drowsy_color, 2)
    cv2.putText(frame, f"Stress: {stress_text} ({stress_score:.2f})", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, stress_color, 2)
    cv2.imshow("Realtime_Final", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

engine.stop()
stream.stop()
cv2.destroyAllWindows()