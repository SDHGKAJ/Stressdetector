import cv2
import numpy as np

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
    if area < 30:
        return None, None, mask
    (x,y),r = cv2.minEnclosingCircle(c)
    return float(r), int(area), mask

def eye_openness_from_roi(eye_roi):
    if eye_roi is None or eye_roi.size==0:
        return None, None
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h,w = eye_roi.shape[:2]
    if not contours:
        return float(h)/float(max(w,1)), None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        x,y,wb,hb = cv2.boundingRect(c)
        if wb==0: return None, None
        ratio = float(hb)/float(wb)
        return ratio, cv2.contourArea(c)
    ellipse = cv2.fitEllipse(c)
    (cx,cy),(MA,ma),angle = ellipse
    if MA==0: return None, None
    ratio = float(ma)/float(MA)
    return ratio, cv2.contourArea(c)

def interocular_distance(face_rect):
    x,y,w,h = face_rect
    return float(w)
