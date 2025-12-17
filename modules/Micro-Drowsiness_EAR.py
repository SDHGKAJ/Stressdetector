import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.25
MICRO_DROWSY_TIME = 0.5  # seconds

eye_close_start = None
micro_drowsy = False

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_ear(eye_points, landmarks, w, h):
    points = []
    for idx in eye_points:
        lm = landmarks[idx]
        points.append(np.array([lm.x * w, lm.y * h]))

    vertical1 = euclidean_distance(points[1], points[5])
    vertical2 = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Start webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(LEFT_EYE, face_landmarks, w, h)
        right_ear = calculate_ear(RIGHT_EYE, face_landmarks, w, h)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if eye_close_start is None:
                eye_close_start = time.time()
            else:
                closed_duration = time.time() - eye_close_start
                if closed_duration >= MICRO_DROWSY_TIME:
                    micro_drowsy = True
        else:
            eye_close_start = None
            micro_drowsy = False

        # Display EAR
        cv2.putText(frame, f"EAR: {ear:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # Status display
        if micro_drowsy:
            cv2.putText(frame, "MICRO-DROWSINESS DETECTED",
                        (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "STATUS: ALERT",
                        (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)

    cv2.imshow("Micro-Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
