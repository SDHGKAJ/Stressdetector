import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

# Ensure the project root is on sys.path so sibling packages like `modules` are importable
# when running this script directly (e.g., `python scripts/realtime_monitor.py`).
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.face_eye_detector import FaceEyeDetector
from modules.feature_extractor import pupil_from_eye_roi, eye_openness_from_roi, interocular_distance

import joblib
import tensorflow as tf

# Safe model loaders
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


def open_camera(indices=(0, 1, 2, 3), backends=(cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)):
    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Opened camera idx={idx} backend={backend}")
                return cap
            cap.release()
    return None


def fmt(v):
    return f"{v:.3f}" if (v is not None and not np.isnan(v)) else "nan"


def main():
    parser = argparse.ArgumentParser(description="Realtime feature collector for stress / cogload / drowsiness models")
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index to open (default: 0)')
    parser.add_argument('--duration', type=float, default=0.0, help='How long to run in seconds (0 = forever)')
    parser.add_argument('--no-display', action='store_true', help='Do not show cv2 windows')
    parser.add_argument('--save-output', type=str, help='CSV path to save collected rows')
    parser.add_argument('--step', type=float, default=0.6, help='Scoring step interval in seconds')
    parser.add_argument('--window', type=float, default=3.0, help='Sliding window size in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Run initialization and print first computed features then exit')
    parser.add_argument('--no-predict', action='store_true', help='Disable model predictions (useful for debugging)')
    args = parser.parse_args()

    detector = FaceEyeDetector()
    cap = open_camera(indices=(args.camera_index,))
    if cap is None or not cap.isOpened():
        raise RuntimeError("ERROR: could not open any camera. Check camera index and permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Limit to 30 FPS to reduce lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency

    # Model paths and loading (respect --no-predict)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    EYE_STATE_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "eye_state_model.keras"))
    YAWN_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "yawn_model.keras"))
    STRESS_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "stress_model.keras"))
    COGLOAD_MODEL_LGB = os.path.normpath(os.path.join(BASE_DIR, "models", "cogload_model_lgb_gpu.joblib"))
    COGLOAD_SCALER_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "cogload_scaler.joblib"))

    if not args.no_predict:
        eye_state_model = _safe_load_model(EYE_STATE_MODEL_PATH, "eye_state")
        yawn_model = _safe_load_model(YAWN_MODEL_PATH, "yawn")
        stress_model = _safe_load_model(STRESS_MODEL_PATH, "stress")
        cog_model = _safe_load_joblib(COGLOAD_MODEL_LGB, "cogload_lgb")
        cog_scaler = _safe_load_joblib(COGLOAD_SCALER_PATH, "cogload_scaler")

        if cog_model is None or cog_scaler is None:
            print("Warning: Cognitive load model or scaler not found; cognitive load predictions disabled.")
    else:
        eye_state_model = None
        yawn_model = None
        stress_model = None
        cog_model = None
        cog_scaler = None

    WINDOW_S = args.window
    STEP_S = args.step
    MICRO_WINDOW = 20
    EYE_CLOSED_THRESHOLD = 0.6
    YAWN_THRESHOLD = 0.6
    EYE_RATIO_LIMIT = 0.6
    YAWN_RATIO_LIMIT = 0.3

    feature_window = deque()
    time_window = deque()
    blink_timestamps = deque()
    radius_smooth = deque(maxlen=6)
    eye_buffer = deque(maxlen=MICRO_WINDOW)
    yawn_buffer = deque(maxlen=MICRO_WINDOW)

    collected = []
    blink_state = False
    last_step = time.time()
    start_time = time.time()

    print("Starting realtime feature collection. Press 'q' in the display window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_now = time.time()

            dets = detector.detect(frame)
            pupil_radii = []
            openness_vals = []
            iol = None
            face_eye_probs = []
            face_yawn_probs = []
            frame_stress_probs = []
            eye_state_preds = []
            yawn_preds = []

            for d in dets:
                fx, fy, fw, fh = d['face']
                face_roi = frame[fy:fy+fh, fx:fx+fw].copy()
                # stress model prediction per-face (if available)
                try:
                    if stress_model is not None:
                        img = cv2.resize(face_roi, (224, 224)).astype(np.float32)
                        img = img / 255.0
                        img = np.expand_dims(img, axis=0)
                        sp = float(stress_model.predict(img, verbose=0)[0][0])
                    else:
                        sp = np.nan
                    frame_stress_probs.append(sp)
                except Exception:
                    pass

                # Yawn prediction
                if not args.no_predict and yawn_model is not None:
                    try:
                        yawn_img = cv2.resize(face_roi, (224, 224)).astype(np.float32)
                        yawn_img = yawn_img / 255.0
                        yawn_img = np.expand_dims(yawn_img, axis=0)
                        yawn_pred_val = float(yawn_model.predict(yawn_img, verbose=0)[0][0])
                        yawn_preds.append(yawn_pred_val > YAWN_THRESHOLD)
                    except Exception:
                        yawn_preds.append(False)
                else:
                    yawn_preds.append(False)

                iol = interocular_distance((fx, fy, fw, fh))
                eyes = d['eyes'][:2]

                for (ex, ey, ew, eh) in eyes:
                    ex2, ey2, ew2, eh2 = ex, ey, ew, eh
                    if ex2 < 0 or ey2 < 0 or ex2 + ew2 > frame.shape[1] or ey2 + eh2 > frame.shape[0]:
                        continue
                    eye_roi = frame[ey2:ey2+eh2, ex2:ex2+ew2].copy()
                    r, area, mask = pupil_from_eye_roi(eye_roi)
                    op, cont_area = eye_openness_from_roi(eye_roi)
                    if r is not None:
                        pupil_radii.append(r)
                    if op is not None:
                        openness_vals.append(op)

                    # Eye state prediction (closed/open) - on first eye only
                    if not args.no_predict and eye_state_model is not None and len(eye_state_preds) == 0:
                        try:
                            eye_img = cv2.resize(eye_roi, (224, 224)).astype(np.float32)
                            eye_img = eye_img / 255.0
                            eye_img = np.expand_dims(eye_img, axis=0)
                            eye_pred = float(eye_state_model.predict(eye_img, verbose=0)[0][0])
                            eye_state_preds.append(eye_pred > EYE_CLOSED_THRESHOLD)
                        except Exception:
                            pass

            # append features to sliding window
            feat = {'pupil_r_px': np.nan, 'openness': np.nan, 'iol_px': np.nan}
            if pupil_radii:
                feat['pupil_r_px'] = float(np.mean(pupil_radii))
            if openness_vals:
                feat['openness'] = float(np.mean(openness_vals))
            if iol:
                feat['iol_px'] = float(iol)

            feature_window.append(feat)
            time_window.append(t_now)

            while len(time_window) > 0 and (t_now - time_window[0]) > (WINDOW_S + 0.5):
                time_window.popleft(); feature_window.popleft()

            # blink detection based on openness mean
            eye_closed_event = False
            if openness_vals:
                op_mean_tmp = float(np.mean(openness_vals))
                if op_mean_tmp < 0.22:
                    if not blink_state:
                        blink_state = True
                        blink_timestamps.append(t_now)
                else:
                    blink_state = False

            # Append predictions to buffers
            eye_state_pred = bool(eye_state_preds[0]) if eye_state_preds else False
            yawn_pred = bool(yawn_preds[0]) if yawn_preds else False

            eye_buffer.append(eye_state_pred)
            yawn_buffer.append(yawn_pred)

            # scoring (run every STEP_S)
            if t_now - last_step >= STEP_S:
                last_step = t_now
                pupil_vals = np.array([f['pupil_r_px'] for f in feature_window if not np.isnan(f['pupil_r_px'])])
                openness_arr = np.array([f['openness'] for f in feature_window if not np.isnan(f['openness'])])
                iol_vals = np.array([f['iol_px'] for f in feature_window if not np.isnan(f['iol_px'])])
                iris_mean = float(pupil_vals.mean()) if pupil_vals.size > 0 else np.nan
                iris_std = float(pupil_vals.std()) if pupil_vals.size > 0 else np.nan
                op_mean = float(openness_arr.mean()) if openness_arr.size > 0 else np.nan
                iol_mean = float(iol_vals.mean()) if iol_vals.size > 0 else np.nan

                while blink_timestamps and (t_now - blink_timestamps[0]) > WINDOW_S:
                    blink_timestamps.popleft()
                blink_rate = float(len(blink_timestamps) / max(0.001, WINDOW_S))

                if not np.isnan(iris_mean):
                    radius_smooth.append(iris_mean)
                iris_mean_sm = float(np.mean(radius_smooth)) if radius_smooth else iris_mean
                iris_norm = iris_mean_sm / iol_mean if (not np.isnan(iris_mean_sm) and not np.isnan(iol_mean) and iol_mean != 0) else np.nan
                baseline_vals = [f['pupil_r_px'] / f['iol_px'] for f in feature_window if not np.isnan(f['pupil_r_px']) and not np.isnan(f['iol_px']) and f['iol_px'] > 0]
                baseline = float(np.median(baseline_vals)) if baseline_vals else np.nan

                micro_drowsy = False
                if len(eye_buffer) == MICRO_WINDOW:
                    eye_ratio = sum(eye_buffer) / MICRO_WINDOW
                    yawn_ratio = sum(yawn_buffer) / MICRO_WINDOW
                    if eye_ratio >= EYE_RATIO_LIMIT or yawn_ratio >= YAWN_RATIO_LIMIT:
                        micro_drowsy = True

                # compute stress probability for this frame (mean across faces)
                stress_prob = float(np.nanmean(frame_stress_probs)) if frame_stress_probs else np.nan

                # Cognitive load prediction (map 4 live features into scaler's expected features)
                cog_pred = None
                cog_prob = None
                cog_scaled_debug = None
                if not args.no_predict and cog_scaler is not None:
                    try:
                        # expected order
                        expected_features = list(getattr(cog_scaler, 'feature_names_in_', ['Pupil_Dilation','Blink_Rate','Fixation_Duration','Saccade_Duration','Speed','Angular_Vel_X','Angular_Vel_Y','Angular_Vel_Z','Steering_Angle','Braking_Response']))
                        mapping = {
                            'Pupil_Dilation': iris_norm,
                            'Blink_Rate': blink_rate,
                            'Fixation_Duration': op_mean,
                            'Saccade_Duration': iris_std,
                        }
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
                        X_scaled = cog_scaler.transform(X_full)
                        cog_scaled_debug = X_scaled.tolist()[0]
                        if cog_model is not None:
                            pred = cog_model.predict(X_scaled)
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
                    except Exception as e:
                        print(f"Warning: cogload processing failed: {e}")

                row = {
                    'ts': t_now,
                    'iris_norm': None if iris_norm != iris_norm else float(iris_norm),
                    'iris_mean_px': None if iris_mean_sm != iris_mean_sm else float(iris_mean_sm),
                    'iris_std_px': iris_std,
                    'iol_mean_px': None if iol_mean != iol_mean else float(iol_mean),
                    'openness_mean': None if op_mean != op_mean else float(op_mean),
                    'blink_rate': blink_rate,
                    'baseline': None if baseline != baseline else float(baseline),
                    'micro_drowsy': bool(micro_drowsy),
                    'stress_prob': None if stress_prob != stress_prob else float(stress_prob),
                    'cogload_pred': None if cog_pred is None else float(cog_pred),
                    'cogload_prob': None if cog_prob is None else float(cog_prob),
                    'cogload_scaled': ','.join([f"{v:.6f}" for v in cog_scaled_debug]) if cog_scaled_debug is not None else ''
                }

                print(f"ts={row['ts']:.3f} iris_norm={fmt(row['iris_norm'])} iris_mean_px={fmt(row['iris_mean_px'])} iris_std_px={fmt(row['iris_std_px'])} iol_mean_px={fmt(row['iol_mean_px'])} openness_mean={fmt(row['openness_mean'])} blink_rate={fmt(row['blink_rate'])} micro_drowsy={row['micro_drowsy']} stress={fmt(row['stress_prob'])} cog={fmt(row['cogload_pred'])} p={fmt(row['cogload_prob'])}")

                collected.append(row)

                # overlays for display - comprehensive metrics
                y_offset = 30
                cv2.putText(frame, f"STRESS: {fmt(row['stress_prob'])}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"CogLoad: {fmt(row['cogload_pred'])}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y_offset += 30
                drowsy_text = "DROWSY!" if row['micro_drowsy'] else "Alert"
                drowsy_color = (0, 0, 255) if row['micro_drowsy'] else (0, 255, 0)
                cv2.putText(frame, f"State: {drowsy_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, drowsy_color, 2)
                y_offset += 30
                cv2.putText(frame, f"Blink Rate: {fmt(row['blink_rate'])}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(frame, f"Openness: {fmt(row['openness_mean'])}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(frame, f"Pupil: {fmt(row['iris_norm'])}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if args.save_output:
                    header = True if not os.path.exists(args.save_output) else False
                    with open(args.save_output, 'a') as f:
                        if header:
                            f.write(','.join(row.keys()) + '\n')
                        f.write(','.join([str(v) for v in row.values()]) + '\n')

                if args.dry_run:
                    print('Dry run complete — exiting')
                    break

            if not args.no_display:
                cv2.imshow('realtime_monitor', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.duration > 0 and (time.time() - start_time) > args.duration:
                print('Duration reached — exiting')
                break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Collected {len(collected)} rows")


if __name__ == '__main__':
    main()
