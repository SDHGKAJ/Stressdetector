import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.face_eye_detector import FaceEyeDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STRESS_MODEL_PATH = os.path.join(BASE_DIR, "Stressdetector", "models", "stress_model.keras")

model = tf.keras.models.load_model(STRESS_MODEL_PATH)

IMG_SIZE = 224
WINDOW = 15  # temporal smoothing
stress_buffer = deque(maxlen=WINDOW)

cap = cv2.VideoCapture(0)
detector = FaceEyeDetector()

if not cap.isOpened():
    raise RuntimeError("Camera not available")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = detector.detect(frame)
    stress_prob = None

    for d in dets:
        fx, fy, fw, fh = d["face"]
        face = frame[fy:fy+fh, fx:fx+fw]
        if face.size == 0:
            continue

        face_img = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_img = face_img.astype(np.float32) / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        stress_prob = float(model.predict(face_img, verbose=0)[0][0])
        break  # only first face

    if stress_prob is not None:
        stress_buffer.append(stress_prob)

    label = "CALIBRATING"
    color = (255, 255, 0)
    avg_stress = 0.0

    if len(stress_buffer) == WINDOW:
        avg_stress = np.mean(stress_buffer)

        if avg_stress < 0.35:
            label = "LOW STRESS"
            color = (0, 255, 0)
        elif avg_stress < 0.65:
            label = "MEDIUM STRESS"
            color = (0, 255, 255)
        else:
            label = "HIGH STRESS"
            color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Stress: {label} ({avg_stress:.2f})",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

    cv2.imshow("Stress Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
