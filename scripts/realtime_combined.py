# scripts/realtime_combined.py
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2, numpy as np, pandas as pd
from collections import deque
import tensorflow as tf
import joblib

from modules.face_eye_detector import FaceEyeDetector
from modules.feature_extractor import pupil_from_eye_roi, eye_openness_from_roi, interocular_distance

# Models (robust paths + safe loads)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EYE_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "eye_state_model.keras"))
YAWN_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "yawn_model.keras"))
STRESS_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "stress_model.keras"))  # match workspace file
# Use the model produced by `train_cognitive_load_model.py`
COGLOAD_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "cognitive_load_model.joblib"))
COGLOAD_MODEL_XGB = os.path.normpath(os.path.join(BASE_DIR, "models", "cogload_model_xgb_gpu.joblib"))
COGLOAD_MODEL_LGB = os.path.normpath(os.path.join(BASE_DIR, "models", "cogload_model_lgb_gpu.joblib"))
# Support both possible scaler filenames produced by different scripts
COGLOAD_SCALER_PATHS = [
    os.path.normpath(os.path.join(BASE_DIR, "models", "cogload_scaler.joblib")),
    os.path.normpath(os.path.join(BASE_DIR, "models", "cognitive_load_scaler.joblib")),
]

def _safe_load_model(path, name):
    try:
        m = tf.keras.models.load_model(path)
        print(f"Loaded {name} model from {path}")
        return m
    except Exception as e:
        print(f"Warning: could not load {name} model from {path}: {e}")
        return None

def _safe_load_joblib(path, name):
    try:
        m = joblib.load(path)
        print(f"Loaded {name} model from {path}")
        return m
    except Exception as e:
        print(f"Warning: could not load {name} joblib model from {path}: {e}")
        return None

eye_model = _safe_load_model(EYE_MODEL_PATH, "eye")
yawn_model = _safe_load_model(YAWN_MODEL_PATH, "yawn")
stress_model = _safe_load_model(STRESS_MODEL_PATH, "stress")
# Prefer the `cognitive_load_model.joblib` produced by your training script; fallback to GPU variants
cog_model = (_safe_load_joblib(COGLOAD_MODEL_PATH, "cognitive_load") or
             _safe_load_joblib(COGLOAD_MODEL_XGB, "cogload_xgb") or
             _safe_load_joblib(COGLOAD_MODEL_LGB, "cogload_lgb"))
# Load scaler if available (support multiple candidate filenames)
cog_scaler = None
for _p in COGLOAD_SCALER_PATHS:
    cog_scaler = _safe_load_joblib(_p, f"cognitive_load_scaler ({os.path.basename(_p)})")
    if cog_scaler is not None:
        break

# Post-load warnings about availability
if cog_model is None and cog_scaler is None:
    print("Warning: No cognitive-load model or scaler found; cognitive load predictions disabled.")
elif cog_model is None:
    print("Warning: Cognitive-load model not found; predictions will be skipped even though scaler is present.")
elif cog_scaler is None:
    print("Warning: Scaler not found; cognitive-load predictions will be skipped even if a model exists.")

# Params
IMG_SIZE = 128
WINDOW_S = 3.0
STEP_S = 0.6
MICRO_WINDOW = 20
EYE_CLOSED_THRESHOLD = 0.6
YAWN_THRESHOLD = 0.6
EYE_RATIO_LIMIT = 0.6
YAWN_RATIO_LIMIT = 0.3

detector = FaceEyeDetector()

def open_camera(indices=(0,1,2,3), backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)):
    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Opened camera idx={idx} backend={backend}")
                return cap
            cap.release()
    return None

cap = open_camera()
if cap is None or not cap.isOpened():
    raise RuntimeError("ERROR: could not open any camera. Close other apps and check Windows Camera Privacy settings.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

feature_window = deque(); time_window = deque()
blink_timestamps = deque()
radius_smooth = deque(maxlen=6)
eye_buffer = deque(maxlen=MICRO_WINDOW)
yawn_buffer = deque(maxlen=MICRO_WINDOW)
collected = []
blink_state = False
last_step = time.time()

def fmt(v):
    return f"{v:.3f}" if (v is not None and not np.isnan(v)) else "nan"

while True:
    ret, frame = cap.read()
    if not ret: break
    t_now = time.time()

    # safe defaults (prevent NameError before first scoring update)
    display_text = "Score:nan  iris_norm:nan  blink/s:nan  open:nan"
    iris_norm = np.nan
    blink_rate = np.nan
    op_mean = np.nan
    stress_prob = np.nan  # ensure defined even if no detections

    dets = detector.detect(frame)
    pupil_radii=[]; openness_vals=[]; iol=None; mask_preview=None
    face_eye_probs = []  # per-face eye probs
    face_yawn_probs = []

    frame_stress_probs = []
    for d in dets:
        fx,fy,fw,fh = d['face']
        face_roi = frame[fy:fy+fh, fx:fx+fw].copy()
        try:
            img = cv2.resize(face_roi, (224,224)).astype(np.float32)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            if stress_model is not None:
                sp = float(stress_model.predict(img, verbose=0)[0][0])
            else:
                sp = np.nan
            frame_stress_probs.append(sp)
        except Exception:
            pass
    # after processing all dets:
    stress_prob = float(np.nanmean(frame_stress_probs)) if frame_stress_probs else np.nan

    for d in dets:
        fx,fy,fw,fh = d['face']
        face_roi = frame[fy:fy+fh, fx:fx+fw].copy()
        iol = interocular_distance((fx,fy,fw,fh))
        eyes = d['eyes'][:2]
        eye_probs = []
        for (ex,ey,ew,eh) in eyes:
            # safety crop
            ex2,ey2,ew2,eh2 = ex,ey,ew,eh
            if ex2<0 or ey2<0 or ex2+ew2>frame.shape[1] or ey2+eh2>frame.shape[0]:
                continue
            eye_roi = frame[ey2:ey2+eh2, ex2:ex2+ew2].copy()
            r, area, mask = pupil_from_eye_roi(eye_roi)
            op, cont_area = eye_openness_from_roi(eye_roi)
            if mask is not None:
                mask_preview = cv2.resize(mask, (160,90))
            if r is not None: pupil_radii.append(r)
            if op is not None: openness_vals.append(op)
            # model input for eye state
            try:
                img = cv2.resize(eye_roi, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                if eye_model is not None:
                    p = float(eye_model.predict(img, verbose=0)[0][0])
                else:
                    p = np.nan
                eye_probs.append(p)
            except Exception:
                pass
            cv2.rectangle(frame, (ex2,ey2), (ex2+ew2, ey2+eh2), (0,255,0), 1)
            if r is not None:
                cx = int(ex2 + ew2/2); cy = int(ey2 + eh2/2); cr = int(max(1, min(int(r), 30)))
                cv2.circle(frame, (cx, cy), cr, (255,0,0), 1)

        # estimate mouth region (lower 40% of face)
        mx1 = fx + int(fw*0.15); mx2 = fx + int(fw*0.85)
        my1 = fy + int(fh*0.55); my2 = fy + int(fh*0.95)
        mx1 = max(0, mx1); mx2 = min(frame.shape[1], mx2); my1 = max(0, my1); my2 = min(frame.shape[0], my2)
        if mx2>mx1 and my2>my1:
            mouth_roi = frame[my1:my2, mx1:mx2].copy()
            try:
                img = cv2.resize(mouth_roi, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                if yawn_model is not None:
                    yp = float(yawn_model.predict(img, verbose=0)[0][0])
                else:
                    yp = np.nan
                face_yawn_probs.append(yp)
                cv2.rectangle(frame, (mx1,my1), (mx2,my2), (255,0,0), 1)
            except Exception:
                pass

        if eye_probs: face_eye_probs.append(np.nanmean(eye_probs))

    # append features to sliding window
    feat = {'pupil_r_px':np.nan, 'openness':np.nan, 'iol_px':np.nan}
    if pupil_radii: feat['pupil_r_px'] = float(np.mean(pupil_radii))
    if openness_vals: feat['openness'] = float(np.mean(openness_vals))
    if iol: feat['iol_px'] = float(iol)
    feature_window.append(feat); time_window.append(t_now)
    while len(time_window)>0 and (t_now - time_window[0]) > (WINDOW_S + 0.5):
        time_window.popleft(); feature_window.popleft()

    # micro-drowsy buffers
    eye_closed_event = False
    yawning_event = False
    if face_eye_probs:
        # Eye model outputs prob of "open" or "closed" depending on training; align with original script:
        eye_prob = np.mean(face_eye_probs)
        eye_closed_event = (eye_prob < EYE_CLOSED_THRESHOLD)
    if face_yawn_probs:
        yawning_event = (np.mean(face_yawn_probs) > YAWN_THRESHOLD)

    eye_buffer.append(eye_closed_event); yawn_buffer.append(yawning_event)
    micro_drowsy = False
    if len(eye_buffer) == MICRO_WINDOW:
        eye_ratio = sum(eye_buffer)/MICRO_WINDOW
        yawn_ratio = sum(yawn_buffer)/MICRO_WINDOW
        if eye_ratio >= EYE_RATIO_LIMIT or yawn_ratio >= YAWN_RATIO_LIMIT:
            micro_drowsy = True

    # scoring (run every STEP_S)
    if t_now - last_step >= STEP_S:
        last_step = t_now
        pupil_vals = np.array([f['pupil_r_px'] for f in feature_window if not np.isnan(f['pupil_r_px'])])
        openness = np.array([f['openness'] for f in feature_window if not np.isnan(f['openness'])])
        iol_vals = np.array([f['iol_px'] for f in feature_window if not np.isnan(f['iol_px'])])
        iris_mean = float(pupil_vals.mean()) if pupil_vals.size>0 else np.nan
        iris_std = float(pupil_vals.std()) if pupil_vals.size>0 else np.nan
        op_mean = float(openness.mean()) if openness.size>0 else np.nan
        iol_mean = float(iol_vals.mean()) if iol_vals.size>0 else np.nan
        while blink_timestamps and (t_now - blink_timestamps[0]) > WINDOW_S:
            blink_timestamps.popleft()
        if not np.isnan(op_mean):
            if op_mean < 0.22:
                if not blink_state:
                    blink_state = True
                    blink_timestamps.append(t_now)
            else:
                blink_state = False
        blink_rate = float(len(blink_timestamps)/max(0.001, WINDOW_S))
        if not np.isnan(iris_mean):
            radius_smooth.append(iris_mean)
        iris_mean_sm = float(np.mean(radius_smooth)) if radius_smooth else iris_mean
        iris_norm = iris_mean_sm / iol_mean if (not np.isnan(iris_mean_sm) and not np.isnan(iol_mean) and iol_mean!=0) else np.nan
        baseline_vals = [f['pupil_r_px']/f['iol_px'] for f in feature_window if not np.isnan(f['pupil_r_px']) and not np.isnan(f['iol_px']) and f['iol_px']>0]
        baseline = float(np.median(baseline_vals)) if baseline_vals else np.nan
        score = 0.5
        # visual stress contribution
        if not np.isnan(stress_prob):
            score += 0.3 * (stress_prob - 0.5)
        if not np.isnan(iris_norm) and not np.isnan(baseline):
            dil = iris_norm - baseline
            score += 2.0 * np.tanh(dil*5.0)
        if not np.isnan(blink_rate):
            if blink_rate < 0.12:
                score += 0.12
            elif blink_rate > 0.25:
                score -= 0.12
        if not np.isnan(op_mean):
            if op_mean < 0.18:
                score += 0.08
        score = max(0.0, min(1.0, score))

        # Cognitive load prediction (requires both model and scaler)
        cog_pred = None
        cog_prob = None
        cog_scaled_debug = None
        try:
            # Live features
            live_feats = {
                'iris_norm': iris_norm,
                'iris_std': iris_std,
                'op_mean': op_mean,
                'blink_rate': blink_rate
            }

            X_scaled = None
            if cog_scaler is not None:
                # Determine expected feature order
                expected_features = None
                if hasattr(cog_scaler, 'feature_names_in_'):
                    expected_features = list(cog_scaler.feature_names_in_)
                else:
                    expected_features = ['Pupil_Dilation','Blink_Rate','Fixation_Duration','Saccade_Duration','Speed','Angular_Vel_X','Angular_Vel_Y','Angular_Vel_Z','Steering_Angle','Braking_Response']

                # Map live features into expected features
                mapping = {
                    'Pupil_Dilation': live_feats['iris_norm'],    # map iris_norm -> Pupil_Dilation
                    'Blink_Rate': live_feats['blink_rate'],
                    'Fixation_Duration': live_feats['op_mean'],   # openness_mean -> Fixation_Duration
                    'Saccade_Duration': live_feats['iris_std'],   # iris_std -> Saccade_Duration
                }

                # Fill values: use mapping if available, otherwise use scaler means if present, else 0.0
                scaler_means = getattr(cog_scaler, 'mean_', None)
                full_vec = []
                for i, fname in enumerate(expected_features):
                    if fname in mapping and mapping[fname] is not None and not (isinstance(mapping[fname], float) and np.isnan(mapping[fname])):
                        full_vec.append(float(mapping[fname]))
                    else:
                        if scaler_means is not None and len(scaler_means) == len(expected_features):
                            full_vec.append(float(scaler_means[i]))
                        else:
                            full_vec.append(0.0)

                X_full = np.array([full_vec])
                try:
                    X_scaled = cog_scaler.transform(X_full)
                    cog_scaled_debug = X_scaled.tolist()[0]
                    # annotate debug about mapping
                    # print once per run could be noisy; leave as debug when scaler present
                except Exception as e:
                    print(f"Warning: scaler transform failed: {e}")
                    X_scaled = None
            else:
                X_scaled = None

            # only predict if both scaler and model are available
            if cog_model is not None and X_scaled is not None:
                pred = cog_model.predict(X_scaled)
                # support array-like or scalar outputs
                try:
                    cog_pred = float(pred[0])
                except Exception:
                    cog_pred = float(pred)
                if hasattr(cog_model, 'predict_proba'):
                    try:
                        probs = cog_model.predict_proba(X_scaled)[0]
                        cog_prob = float(np.max(probs))
                    except Exception:
                        cog_prob = None
            else:
                # no prediction due to missing model or scaler
                pass
        except Exception as e:
            print(f"Warning: cogload processing failed: {e}")

        # build display text and include scaled features if available
        display_text = f"Score:{fmt(score)}  iris_norm:{fmt(iris_norm)}  blink/s:{fmt(blink_rate)}  open:{fmt(op_mean)}  cog:{fmt(cog_pred) if cog_pred is not None else 'nan'} p:{fmt(cog_prob) if cog_prob is not None else 'nan'}"
        if cog_scaled_debug is not None:
            try:
                scaled_str = ','.join([f"{v:.3f}" for v in cog_scaled_debug])
                display_text += f"  scaled:[{scaled_str}]"
            except Exception:
                pass

        display_text = f"Score:{fmt(score)}  iris_norm:{fmt(iris_norm)}  blink/s:{fmt(blink_rate)}  open:{fmt(op_mean)}  cog:{fmt(cog_pred) if cog_pred is not None else 'nan'} p:{fmt(cog_prob) if cog_prob is not None else 'nan'}"
        print(display_text)
        row = {'ts':t_now,'iris_norm':None if iris_norm!=iris_norm else float(iris_norm),'iris_mean_px':None if iris_mean_sm!=iris_mean_sm else float(iris_mean_sm),'iris_std_px':iris_std,'iol_mean_px':None if iol_mean!=iol_mean else float(iol_mean),'openness_mean':None if op_mean!=op_mean else float(op_mean),'blink_rate':blink_rate,'score':score,'cogload_pred':None if cog_pred is None else float(cog_pred),'cogload_prob':None if cog_prob is None else float(cog_prob)}
        collected.append(row)

    # overlays
        # overlays
    cv2.putText(frame, display_text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame,
                f"Eye closed prob: {1 - (np.mean(face_eye_probs) if face_eye_probs else 0):.2f}",
                (20,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

    cv2.putText(frame,
                f"Yawn prob: {np.mean(face_yawn_probs) if face_yawn_probs else 0:.2f}",
                (20,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

    cv2.putText(frame,
                f"Stress prob: {stress_prob:.2f}" if stress_prob==stress_prob else "Stress prob: nan",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)

    cog_overlay = f"CogLoad: {fmt(cog_pred) if cog_pred is not None else 'nan'} p:{fmt(cog_prob) if cog_prob is not None else 'nan'}"
    cv2.putText(frame,
                cog_overlay,
                (20,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255),2)

    if micro_drowsy:
        cv2.putText(frame, "MICRO-DROWSINESS DETECTED",
                    (30,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
    else:
        cv2.putText(frame, "STATUS: ALERT",
                    (30,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

    if mask_preview is not None:
        cv2.imshow('pupil_mask', mask_preview)

    cv2.imshow("Combined - Cognitive Load & Micro-Drowsy", frame)

cap.release(); cv2.destroyAllWindows()