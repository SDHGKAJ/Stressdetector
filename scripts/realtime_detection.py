import cv2
import time
import numpy as np
import pandas as pd
from collections import deque

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

def detect_faces_eyes(frame, sf=1.05, mn=4, ms=(60,60)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=ms)
    res = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20,10))
        eyes_global = []
        for (ex,ey,ew,eh) in eyes:
            eyes_global.append((x+ex, y+ey, ew, eh))
        res.append({'face':(x,y,w,h), 'eyes':eyes_global})
    return res

def pupil_from_eye_roi(eye_roi):
    if eye_roi is None or eye_roi.size==0:
        return None, None, None
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = th.copy()
    if not contours:
        return None, None, mask
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 20:
        return None, None, mask
    (x,y),r = cv2.minEnclosingCircle(c)
    return float(r), int(area), mask

def eye_openness_from_roi(eye_roi):
    if eye_roi is None or eye_roi.size == 0:
        return None, None
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 3)
    h, w = th.shape
    band_h = max(6, h//3)
    y0 = h//3
    band = th[y0:y0+band_h, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        dark_ratio = 1.0 - (np.count_nonzero(band) / float(band.size))
        openness = np.clip(1.0 - dark_ratio, 0.0, 1.0)
        return openness, band
    c = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    gap_height = bh
    openness = float(gap_height) / float(band_h)
    openness = np.clip(openness, 0.0, 1.0)
    return openness, band

def interocular_distance(face_rect):
    x,y,w,h = face_rect
    return float(w)

def fmt(v):
    return f"{v:.3f}" if (v is not None and not (v!=v)) else "nan"

# Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parameters
calib_secs = 10.0
window_s = 3.0
step_s = 0.5

feature_window = deque()
time_window = deque()
collected = []
radius_smooth = deque(maxlen=8)
op_smooth = deque(maxlen=8)

# Blink detection state
BLINK_STATE = False
blink_count_window = deque()
MIN_CLOSED_SECS = 0.04            # shorter min to catch quick blinks
last_closed_time = None

# Pupil-missing fallback settings
pupil_missing_count = 0
PUPIL_MISSING_FRAMES_TO_CLOSE = 1  # be sensitive
pupil_reappear_count = 0
PUPIL_AREA_CLOSE_THRESH = 25      # area below this treated as missing

# Calibration buffers
calib_start = time.time()
calib_iris = []
calib_op = []

last_step = time.time()
mask_preview = None

print("Calibration: stay neutral and look at camera for {:.0f} seconds...".format(calib_secs))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_now = time.time()
        dets = detect_faces_eyes(frame)
        pupil_radii = []
        pupil_areas = []
        openness_vals = []
        iol = None
        mask_preview = None
        feat = {'pupil_r_px': np.nan, 'openness': np.nan, 'iol_px': np.nan}

        for d in dets:
            fx,fy,fw,fh = d['face']
            iol = interocular_distance((fx,fy,fw,fh))
            eyes = d['eyes'][:2]
            for (ex,ey,ew,eh) in eyes:
                ex2,ey2,ew2,eh2 = ex,ey,ew,eh
                if ex2 < 0 or ey2 < 0 or ex2+ew2 > frame.shape[1] or ey2+eh2 > frame.shape[0]:
                    continue
                eye_roi = frame[ey2:ey2+eh2, ex2:ex2+ew2].copy()
                r, area, mask = pupil_from_eye_roi(eye_roi)
                op, band = eye_openness_from_roi(eye_roi)
                if mask is not None:
                    try:
                        mask_preview = cv2.resize(mask, (160,90))
                    except:
                        mask_preview = None
                if r is not None:
                    pupil_radii.append(r)
                if area is not None:
                    pupil_areas.append(area)
                if op is not None:
                    openness_vals.append(op)
                cv2.rectangle(frame, (ex2,ey2), (ex2+ew2, ey2+eh2), (0,255,0), 1)
                if r:
                    cx = int(ex2 + ew2/2)
                    cy = int(ey2 + eh2/2)
                    cr = int(max(1, min(int(r), 30)))
                    cv2.circle(frame, (cx, cy), cr, (255,0,0), 1)

        if pupil_radii:
            feat['pupil_r_px'] = float(np.mean(pupil_radii))
        if openness_vals:
            feat['openness'] = float(np.mean(openness_vals))
        if iol:
            feat['iol_px'] = float(iol)

        feature_window.append(feat)
        time_window.append(t_now)

        while len(time_window) > 0 and (t_now - time_window[0]) > (window_s + 0.5):
            time_window.popleft(); feature_window.popleft()

        if t_now - last_step >= step_s:
            last_step = t_now

            pupil_vals = np.array([f['pupil_r_px'] for f in feature_window if not np.isnan(f['pupil_r_px'])])
            openness = np.array([f['openness'] for f in feature_window if not np.isnan(f['openness'])])
            iol_vals = np.array([f['iol_px'] for f in feature_window if not np.isnan(f['iol_px'])])

            iris_mean = float(pupil_vals.mean()) if pupil_vals.size>0 else np.nan
            iris_std = float(pupil_vals.std()) if pupil_vals.size>0 else np.nan
            op_mean = float(openness.mean()) if openness.size>0 else np.nan
            iol_mean = float(iol_vals.mean()) if iol_vals.size>0 else np.nan

            # calibration
            if (t_now - calib_start) <= calib_secs:
                if not np.isnan(iris_mean):
                    calib_iris.append(iris_mean)
                if not np.isnan(op_mean):
                    calib_op.append(op_mean)
                elapsed = t_now - calib_start
                cv2.putText(frame, f"Calibrating {int(max(0, calib_secs-elapsed))}s", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

            # adaptive baseline & thresholds
            if (t_now - calib_start) > calib_secs and len(calib_iris)>0 and len(calib_op)>0:
                base_iris = float(np.median(calib_iris))
                base_op = float(np.median(calib_op))
                # use relative-drop threshold: closed when base_op - op > delta_op
                DELTA_OP_CLOSED = 0.18   # if openness drops by >18% of scale -> closed
                # also keep some absolute fallback
                CLOSED_TH_ABS = max(0.12, base_op * 0.6)
            else:
                base_iris = np.nan
                base_op = np.nan
                DELTA_OP_CLOSED = 0.18
                CLOSED_TH_ABS = 0.28

            # smoothing
            if not np.isnan(op_mean):
                op_smooth.append(op_mean)
            if not np.isnan(iris_mean):
                radius_smooth.append(iris_mean)

            op_mean_sm = float(np.mean(op_smooth)) if op_smooth else op_mean
            iris_mean_sm = float(np.mean(radius_smooth)) if radius_smooth else iris_mean

            # pupil-missing bookkeeping: look at latest pupil area if available
            last_area = None
            if pupil_areas:
                last_area = max(pupil_areas)  # take max area detected in frame
            # update missing counters
            if last_area is None or last_area < PUPIL_AREA_CLOSE_THRESH:
                pupil_missing_count += 1
                pupil_reappear_count = 0
            else:
                pupil_missing_count = 0
                pupil_reappear_count += 1

            # decide closed by openness: relative drop OR absolute small value
            is_closed_by_openness = False
            if not np.isnan(base_op) and not np.isnan(op_mean_sm):
                delta = base_op - op_mean_sm
                is_closed_by_openness = (delta >= DELTA_OP_CLOSED) or (op_mean_sm < CLOSED_TH_ABS)
            elif not np.isnan(op_mean_sm):
                is_closed_by_openness = op_mean_sm < CLOSED_TH_ABS

            # decide closed by pupil missing
            is_closed_by_pupil_missing = (pupil_missing_count >= PUPIL_MISSING_FRAMES_TO_CLOSE)

            currently_closed = is_closed_by_openness or is_closed_by_pupil_missing

            now = t_now
            # state machine
            if currently_closed:
                if not BLINK_STATE:
                    BLINK_STATE = True
                    last_closed_time = now
            else:
                if BLINK_STATE:
                    closed_duration = now - (last_closed_time or now)
                    BLINK_STATE = False
                    last_closed_time = None
                    if closed_duration >= MIN_CLOSED_SECS or is_closed_by_pupil_missing:
                        blink_count_window.append(now)

            # cleanup old blink timestamps
            while blink_count_window and (now - blink_count_window[0]) > window_s:
                blink_count_window.popleft()

            blink_rate = float(len(blink_count_window)/max(0.001, window_s))

            # pupil normalization
            iris_norm = iris_mean_sm / iol_mean if (not np.isnan(iris_mean_sm) and not np.isnan(iol_mean) and iol_mean!=0) else np.nan

            baseline_vals = [f['pupil_r_px']/f['iol_px'] for f in feature_window if not np.isnan(f['pupil_r_px']) and not np.isnan(f['iol_px']) and f['iol_px']>0]
            baseline = float(np.median(baseline_vals)) if baseline_vals else np.nan

            # heuristic cognitive load score
            score = 0.5
            if not np.isnan(iris_norm) and not np.isnan(baseline):
                dil = iris_norm - baseline
                score += 2.0 * np.tanh(dil*5.0)
            if not np.isnan(blink_rate):
                if blink_rate < 0.12:
                    score += 0.12
                elif blink_rate > 0.25:
                    score -= 0.12
            if not np.isnan(op_mean_sm):
                if is_closed_by_openness:
                    score += 0.08
            score = max(0.0, min(1.0, score))

            # label and display
            level = "LOW"
            color = (0,200,0)
            if score >= 0.7:
                level = "HIGH"; color = (0,0,200)
            elif score >= 0.4:
                level = "MED"; color = (0,180,180)

            cv2.putText(frame, f"{level}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            display_text = f"Score:{fmt(score)} iris_norm:{fmt(iris_norm)} pupil:{fmt(iris_mean_sm)} op:{fmt(op_mean_sm)} blink/s:{fmt(blink_rate)}"
            cv2.putText(frame, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            print(display_text,
                  f"BASE_iris:{fmt(base_iris)} BASE_op:{fmt(base_op)} DELTA_OP_CLOSED:{fmt(DELTA_OP_CLOSED)} CLOSED_ABS:{fmt(CLOSED_TH_ABS)} PupArea:{fmt(last_area)} Pupil_missing:{pupil_missing_count}")

            row = {
                'ts': now,
                'iris_norm': None if iris_norm!=iris_norm else float(iris_norm),
                'iris_mean_px': None if iris_mean_sm!=iris_mean_sm else float(iris_mean_sm),
                'iris_std_px': iris_std,
                'iol_mean_px': None if iol_mean!=iol_mean else float(iol_mean),
                'openness_mean': None if op_mean_sm!=op_mean_sm else float(op_mean_sm),
                'blink_rate': blink_rate,
                'score': score,
                'level': level
            }
            collected.append(row)

        # show mask preview if available
        if mask_preview is not None:
            cv2.imshow('pupil_mask', mask_preview)

        cv2.imshow('Cognitive Load - OpenCV (Adaptive)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('s'):
            try:
                pd.DataFrame(collected).to_csv('session_features.csv', index=False)
                print('Saved session_features.csv')
            except Exception as e:
                print('Save error', e)
        if key in (ord('l'), ord('m'), ord('h')) and collected:
            lab = {ord('l'):'low', ord('m'):'med', ord('h'):'high'}[key]
            df = pd.DataFrame(collected)
            df['label'] = lab
            fname = f"dataset_{lab}_{int(time.time())}.csv"
            df.to_csv(fname, index=False)
            print('Saved', fname)
            collected = []

finally:
    cap.release()
    cv2.destroyAllWindows()
