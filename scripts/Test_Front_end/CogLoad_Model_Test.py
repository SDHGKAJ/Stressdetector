import cv2
import numpy as np
import joblib
import time
from collections import deque
import pandas as pd
score_buffer = deque(maxlen=15)


model = joblib.load("models/cogload_model_lgb_gpu.joblib")
scaler = joblib.load("models/cogload_scaler.joblib")

FEATURE_ORDER = [
    'Pupil_Dilation',
    'Blink_Rate',
    'Fixation_Duration',
    'Saccade_Duration'
]

try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe installed but compiled components missing")
except Exception as e:
    raise RuntimeError(
        "MediaPipe is not usable in this environment.\n"
        "Use Python 3.10–3.11 and reinstall mediapipe.\n"
        f"Original error: {e}"
    )

score_buffer = deque(maxlen=15)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

blink_history = deque(maxlen=30)
fixation_start = None
last_gaze = None
saccade_durations = deque(maxlen=10)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def cogload_to_label(score):
    if score < 0.50:
        return "LOW", (0, 255, 0)
    elif score < 1.00:
        return "MEDIUM", (0, 255, 255)
    else:
        return "HIGH", (0, 0, 255)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    pupil_dilation = blink_rate = fixation_duration = saccade_duration = 0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        left_eye_idx = [33, 160, 158, 133, 153, 144]
        left_eye = np.array([(lm[i].x, lm[i].y) for i in left_eye_idx])
        ear = eye_aspect_ratio(left_eye)

        blink = 1 if ear < 0.2 else 0
        blink_history.append(blink)
        blink_rate = sum(blink_history)

        pupil_dilation = np.linalg.norm(left_eye[1] - left_eye[4])

        gaze = np.array([lm[468].x, lm[468].y])

        if last_gaze is not None:
            movement = np.linalg.norm(gaze - last_gaze)

            if movement < 0.002:
                if fixation_start is None:
                    fixation_start = time.time()
                fixation_duration = time.time() - fixation_start
            else:
                if fixation_start is not None:
                    saccade_durations.append(time.time() - fixation_start)
                fixation_start = None

        last_gaze = gaze
        saccade_duration = np.mean(saccade_durations) if saccade_durations else 0

        features = np.array([[
            pupil_dilation,
            blink_rate,
            fixation_duration,
            saccade_duration
        ]])

        features = pd.DataFrame([[pupil_dilation, blink_rate, fixation_duration, saccade_duration]],
                        columns=FEATURE_ORDER)
        features_scaled = scaler.transform(features)
        scaled_df = pd.DataFrame(features_scaled, columns=FEATURE_ORDER)
        score_buffer.append(model.predict(scaled_df)[0])
        cogload = np.mean(score_buffer)

        label, color = cogload_to_label(cogload)

        cv2.putText(
            frame,
            f"Cognitive Load: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Score: {cogload:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Cognitive Load Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
