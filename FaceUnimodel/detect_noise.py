import cv2
import numpy as np
import imutils
import sys
import mediapipe as mp


if len(sys.argv) > 1:
    path = sys.argv[1]

# detecting ears
# basic tree based volia-jones, doesn't seem to work for non-frontal images
def detect_haarcascade(path):
    left_ear_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mcs_leftear.xml')
    right_ear_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mcs_rightear.xml')
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface.xml')
    lefteye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_lefteye.xml')
    righteye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_righteye.xml')
    # eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

    img = cv2.imread(path)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 1)
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 1)
    face = face_cascade.detectMultiScale(gray, 1.3 , 5)
    left_eye = lefteye_cascade.detectMultiScale(gray, 1.5 , 5)
    right_eye = righteye_cascade.detectMultiScale(gray, 1.5 , 5)

    for (x,y,w,h) in left_ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

    for (x,y,w,h) in right_ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
    
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
    
    for (x,y,w,h) in left_eye:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)


    for (x,y,w,h) in right_eye:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

    cv2.imwrite('../dumps/example_har.png', img)

# media pipe with face landmarks

if __name__ == "__main__":
    if path:
        detect_haarcascade(path)
