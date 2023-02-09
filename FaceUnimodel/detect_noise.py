import cv2
import numpy as np
import imutils
import sys
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

path = None
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
def detect_mediapipe(path):
    # Left eye indices list
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    # Right eye indices list
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.6,
                            min_tracking_confidence=0.6) as face_mesh:
            img = cv2.imread(path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            img_h, img_w = img.shape[:2]
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            cv2.polylines(img, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.imwrite('../dumps/example_mediapipe.png', img)
            

if __name__ == "__main__":
    if path:
        # detect_haarcascade(path)
        detect_mediapipe(path)
