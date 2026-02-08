import cv2
import numpy as np
import tensorflow as tf
import joblib
import time
import os
import sys
import pandas as pd
from collections import deque

# ---------------- PATH SETUP ----------------

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from modules.face_eye_detector import FaceEyeDetector

# ---------------- MODELS ----------------

# Cognitive Load

cog_model = joblib.load("models/cogload_model_lgb_gpu.joblib")
cog_scaler = joblib.load("models/cogload_scaler.joblib")

# Micro-drowsiness

eye_model = tf.keras.models.load_model("models/eye_state_model.keras")
yawn_model = tf.keras.models.load_model("models/yawn_model.keras")

# Stress

stress_model = tf.keras.models.load_model("models/stress_model.keras")

# ---------------- PARAMETERS ----------------

IMG_SIZE = 224
WINDOW = 15
EYE_CLOSED_TH = 0.6
YAWN_TH = 0.6
FEATURE_NAMES = [
    "Pupil_Dilation",
    "Blink_Rate",
    "Fixation_Duration",
    "Saccade_Duration"
]

# ---------------- BUFFERS ----------------

cog_buffer = deque(maxlen=15)
stress_buffer = deque(maxlen=WINDOW)
eye_buffer = deque(maxlen=12)
yawn_buffer = deque(maxlen=20)
blink_history = deque(maxlen=30)
saccade_durations = deque(maxlen=10)

# ---------------- MEDIAPIPE ----------------

import mediapipe as mp
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# ---------------- HELPERS ----------------

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)
def cog_label(score):
    if score < 0.5:
        return "LOW", (0,0,255)
    elif score < 1.0:
        return "MEDIUM", (0,255,255)
    return "HIGH", (0,255,0)
def stress_label(score):
    if score < 0.35:
        return "LOW", (0,255,0)
    elif score < 0.65:
        return "MEDIUM", (0,255,255)
    return "HIGH", (0,0,255)

# ---------------- INIT ----------------

cap = cv2.VideoCapture(0)
detector = FaceEyeDetector()
fixation_start = None
last_gaze = None

# ================= MAIN LOOP =================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(rgb)
    face_dets = detector.detect(frame)

    # ---------- COGNITIVE LOAD ----------

    cog_score = 0
    if mesh_results.multi_face_landmarks:
        lm = mesh_results.multi_face_landmarks[0].landmark
        left_eye_idx = [33,160,158,133,153,144]
        left_eye = np.array([(lm[i].x, lm[i].y) for i in left_eye_idx])
        ear = eye_aspect_ratio(left_eye)
        blink = int(ear < 0.2)
        blink_history.append(blink)
        pupil_dilation = np.linalg.norm(left_eye[1] - left_eye[4])
        blink_rate = sum(blink_history)
        gaze = np.array([lm[468].x, lm[468].y])
        fixation_duration = 0
        if last_gaze is not None:
            movement = np.linalg.norm(gaze - last_gaze)
            if movement < 0.002:
                fixation_start = fixation_start or time.time()
                fixation_duration = time.time() - fixation_start
            else:
                if fixation_start:
                    saccade_durations.append(time.time() - fixation_start)
                fixation_start = None
        last_gaze = gaze
        saccade_duration = np.mean(saccade_durations) if saccade_durations else 0
        features = pd.DataFrame([{
            "Pupil_Dilation": pupil_dilation,
            "Blink_Rate": blink_rate,
            "Fixation_Duration": fixation_duration,
            "Saccade_Duration": saccade_duration
        }])
        scaled = cog_scaler.transform(features)
        cog_buffer.append(cog_model.predict(scaled)[0])
        cog_score = np.mean(cog_buffer)
    cog_text, cog_color = cog_label(cog_score)

    # ---------- DROWSINESS + STRESS ----------

    micro_score = 0
    stress_score = 0
    if face_dets:
        fx, fy, fw, fh = face_dets[0]["face"]
        face = frame[fy:fy+fh, fx:fx+fw]
        if face.size > 0:
            # Stress
            face_img = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
            face_img = np.expand_dims(face_img.astype(np.float32), 0)
            stress_prob = stress_model.predict(face_img, verbose=0)[0][0]
            stress_buffer.append(stress_prob)
            stress_score = np.mean(stress_buffer)

            # Yawn

            yawn_img = cv2.resize(face, (128,128)) / 255.0
            yawn_img = np.expand_dims(yawn_img.astype(np.float32), 0)
            yawn = yawn_model.predict(yawn_img, verbose=0)[0][0] > YAWN_TH
            yawn_buffer.append(int(yawn))

            # Eye
            
            eyes = face_dets[0].get("eyes", [])
            eye_closed_votes = []
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(
                    frame,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (0, 255, 0),   # green
                    2
                )
                eye = frame[ey:ey+eh, ex:ex+ew]
                if eye.size > 0:
                    eye_img = cv2.resize(eye, (128,128)) / 255.0
                    eye_img = np.expand_dims(eye_img.astype(np.float32), 0)

                    prob = eye_model.predict(eye_img, verbose=0)[0][0]
                    eye_closed_votes.append(prob > EYE_CLOSED_TH)
            if eye_closed_votes:
                eye_buffer.append(int(np.mean(eye_closed_votes) > 0.5))
            else:
                eye_buffer.append(0)
        if len(eye_buffer) == eye_buffer.maxlen:
            eye_ratio = sum(eye_buffer) / len(eye_buffer)
            yawn_ratio = sum(yawn_buffer) / max(len(yawn_buffer), 1)
            micro_score = max(eye_ratio, 0.7 * yawn_ratio)

        
            print(
                f"EyeRatio:{eye_ratio:.2f}  "
                f"YawnRatio:{yawn_ratio:.2f}  "
                f"Micro:{micro_score:.2f}"
            )
    stress_text, stress_color = stress_label(stress_score)
    DROWSY_TH = 0.5
    drowsy_text = "DROWSY" if micro_score > DROWSY_TH else "ALERT"
    drowsy_color = (0,0,255) if micro_score > DROWSY_TH else (0,255,0)

    # ---------- DISPLAY ----------

    cv2.putText(frame, f"CogLoad: {cog_text} ({cog_score:.2f})", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cog_color, 2)
    cv2.putText(frame, f"Drowsy: {drowsy_text} ({micro_score:.2f})", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, drowsy_color, 2)
    cv2.putText(frame, f"Stress: {stress_text} ({stress_score:.2f})", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, stress_color, 2)
    cv2.imshow("Realtime_Final", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
