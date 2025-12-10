import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import time
import numpy as np
import pandas as pd
from collections import deque
from modules.face_eye_detector import FaceEyeDetector
from modules.feature_extractor import pupil_from_eye_roi, eye_openness_from_roi, interocular_distance

detector = FaceEyeDetector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window_s = 3.0
step_s = 0.6
feature_window = deque()
time_window = deque()
blink_timestamps = deque()
collected = []
blink_state = False
last_step = time.time()
radius_smooth = deque(maxlen=6)
mask_preview = None

def fmt(v):
    return f"{v:.3f}" if (v is not None and not (v!=v)) else "nan"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t_now = time.time()
    dets = detector.detect(frame)
    pupil_radii = []
    openness_vals = []
    iol = None
    mask_preview = None
    for d in dets:
        fx,fy,fw,fh = d['face']
        iol = interocular_distance((fx,fy,fw,fh))
        eyes = d['eyes'][:2]
        for (ex,ey,ew,eh) in eyes:
            ex2,ey2 = ex,ey
            ew2,eh2 = ew,eh
            if ex2<0 or ey2<0 or ex2+ew2>frame.shape[1] or ey2+eh2>frame.shape[0]:
                continue
            eye_roi = frame[ey2:ey2+eh2, ex2:ex2+ew2].copy()
            r, area, mask = pupil_from_eye_roi(eye_roi)
            op, cont_area = eye_openness_from_roi(eye_roi)
            if mask is not None:
                mask_preview = cv2.resize(mask, (160,90))
            if r is not None:
                pupil_radii.append(r)
            if op is not None:
                openness_vals.append(op)
            cv2.rectangle(frame, (ex2,ey2), (ex2+ew2, ey2+eh2), (0,255,0), 1)
            if r:
                cx = int(ex2 + ew2/2)
                cy = int(ey2 + eh2/2)
                cr = int(max(1, min(int(r), 30)))
                cv2.circle(frame, (cx, cy), cr, (255,0,0), 1)
    feat = {'pupil_r_px':np.nan, 'openness':np.nan, 'iol_px':np.nan}
    if pupil_radii:
        feat['pupil_r_px'] = float(np.mean(pupil_radii))
    if openness_vals:
        feat['openness'] = float(np.mean(openness_vals))
    if iol:
        feat['iol_px'] = float(iol)
    feature_window.append(feat)
    time_window.append(t_now)
    while len(time_window)>0 and (t_now - time_window[0]) > (window_s + 0.5):
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
        while blink_timestamps and (t_now - blink_timestamps[0]) > window_s:
            blink_timestamps.popleft()
        if not np.isnan(op_mean):
            if op_mean < 0.22:
                if not blink_state:
                    blink_state = True
                    blink_timestamps.append(t_now)
            else:
                blink_state = False
        blink_rate = float(len(blink_timestamps)/max(0.001, window_s))
        if not np.isnan(iris_mean):
            radius_smooth.append(iris_mean)
        iris_mean_sm = float(np.mean(radius_smooth)) if radius_smooth else iris_mean
        iris_norm = iris_mean_sm / iol_mean if (not np.isnan(iris_mean_sm) and not np.isnan(iol_mean) and iol_mean!=0) else np.nan
        baseline_vals = [f['pupil_r_px']/f['iol_px'] for f in feature_window if not np.isnan(f['pupil_r_px']) and not np.isnan(f['iol_px']) and f['iol_px']>0]
        baseline = float(np.median(baseline_vals)) if baseline_vals else np.nan
        score = 0.5
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
        display_text = f"Score:{fmt(score)}  iris_norm:{fmt(iris_norm)}  blink/s:{fmt(blink_rate)}  open:{fmt(op_mean)}"
        cv2.putText(frame, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        print(display_text)
        row = {'ts':t_now,'iris_norm':None if iris_norm!=iris_norm else float(iris_norm),'iris_mean_px':None if iris_mean_sm!=iris_mean_sm else float(iris_mean_sm),'iris_std_px':iris_std,'iol_mean_px':None if iol_mean!=iol_mean else float(iol_mean),'openness_mean':None if op_mean!=op_mean else float(op_mean),'blink_rate':blink_rate,'score':score}
        collected.append(row)
    if mask_preview is not None:
        cv2.imshow('pupil_mask', mask_preview)
    cv2.imshow('Cognitive Load - OpenCV (Haar)', frame)
    key = cv2.waitKey(1) & 0xFF
    if key==27:
        break
    if key==ord('s'):
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
cap.release()
cv2.destroyAllWindows()
