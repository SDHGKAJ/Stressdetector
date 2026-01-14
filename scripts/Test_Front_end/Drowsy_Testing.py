import cv2
import numpy as np
from collections import deque
import os
import sys
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.face_eye_detector import FaceEyeDetector
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

EYE_MODEL_PATH = os.path.join(BASE_DIR, "models", "eye_state_model.keras")
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.keras")


eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

print("Eye model input:", eye_model.input_shape)
print("Yawn model input:", yawn_model.input_shape)

# ---- MODEL INPUT SPECS ----
EYE_INPUT_SIZE = eye_model.input_shape[-1]          # 57600
YAWN_H, YAWN_W = yawn_model.input_shape[1:3]         # (128, 128)

print("Eye model expects:", EYE_INPUT_SIZE)
print("Yawn model expects:", (YAWN_H, YAWN_W, 3))

print("Eye & Yawn models loaded")
WINDOW = 20          # frames (~0.6 sec @ 30fps)
EYE_CLOSED_TH = 0.6
YAWN_TH = 0.6

EYE_RATIO_LIMIT = 0.6  # 60% closed
YAWN_RATIO_LIMIT = 0.3   # 30% yawning

eye_buffer = deque(maxlen=WINDOW)
yawn_buffer = deque(maxlen=WINDOW)
cap = cv2.VideoCapture(0)
detector = FaceEyeDetector()

if not cap.isOpened():
    raise RuntimeError("Camera not available")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = detector.detect(frame)

    eye_closed = False
    yawn = False

    for d in dets:
        fx, fy, fw, fh = d['face']
        face = frame[fy:fy+fh, fx:fx+fw]

        if face.size == 0:
            continue

        # ---- YAWN PREDICTION (CNN MODEL) ----
        face_img = cv2.resize(face, (YAWN_W, YAWN_H))
        face_img = face_img.astype(np.float32) / 255.0
        face_img = np.expand_dims(face_img, axis=0)  # (1, 128, 128, 3)

        yawn_prob = float(yawn_model.predict(face_img, verbose=0)[0][0])
        yawn = yawn_prob > YAWN_TH


        # ---- EYE STATE PREDICTION (CNN MODEL) ----
        eye_closed = False

        eyes = d.get('eyes', [])

        if eyes:
            ex, ey, ew, eh = eyes[0]
            eye = frame[ey:ey+eh, ex:ex+ew]

            if eye is not None and eye.size > 0:
                EH, EW = eye_model.input_shape[1:3]  # (128, 128)

                eye_img = cv2.resize(eye, (EW, EH))
                eye_img = eye_img.astype(np.float32) / 255.0
                eye_img = np.expand_dims(eye_img, axis=0)  # (1, 128, 128, 3)

                eye_prob = float(eye_model.predict(eye_img, verbose=0)[0][0])
                eye_closed = eye_prob > EYE_CLOSED_TH


        break  # only first detected face
    eye_buffer.append(eye_closed)
    yawn_buffer.append(yawn)

    micro_drowsy_score = 0.0
    micro_drowsy = False

    if len(eye_buffer) == WINDOW:
        eye_ratio = sum(eye_buffer) / WINDOW
        yawn_ratio = sum(yawn_buffer) / WINDOW
        micro_drowsy_score = max(eye_ratio, yawn_ratio)
        micro_drowsy_score = round(micro_drowsy_score, 2)

        if eye_ratio >= EYE_RATIO_LIMIT or yawn_ratio >= YAWN_RATIO_LIMIT:
            micro_drowsy = True
    label = "DROWSY" if micro_drowsy else "ALERT"
    color = (0, 0, 255) if micro_drowsy else (0, 255, 0)
    cv2.putText(
    frame,
    f"Drowsy Score: {micro_drowsy_score:.2f} ({label})",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    color,
    2
    )
    cv2.imshow("Drowsy Testing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
