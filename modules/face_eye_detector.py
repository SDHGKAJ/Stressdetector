import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

class FaceEyeDetector:
    def __init__(self, scaleFactor=1.05, minNeighbors=4, minSize=(60,60)):
        self.sf = scaleFactor
        self.mn = minNeighbors
        self.ms = minSize

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=self.sf, minNeighbors=self.mn, minSize=self.ms)
        results = []
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(24,12))
            eyes_global = []
            for (ex,ey,ew,eh) in eyes:
                eyes_global.append((x+ex, y+ey, ew, eh))
            results.append({'face':(x,y,w,h), 'eyes':eyes_global})
        return results
