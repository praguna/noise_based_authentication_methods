import cv2
import numpy as np
import sys
import mediapipe as mp
from PIL import Image
import itertools
from hair_segmentation import *
import torch
# from torchsr.models import ninasr_b0
from torchvision import transforms
import torchvision.models as models
# mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
resnet_50 = models.resnet50(pretrained=True)
# print(mobilenet_v3_small)
# mobilenet_v3_rep = torch.nn.Sequential(*(list(mobilenet_v3_small.children())[:-1]))
# until_last_layer = torch.nn.Sequential(*(list(mobilenet_v3_small.classifier.children())[:-1]))
# mobilenet_v3_rep = torch.nn.Sequential(mobilenet_v3_rep, torch.nn.Flatten(),  until_last_layer).eval()
resnet_50 =  torch.nn.Sequential(*(list(resnet_50.children())[:-1])).eval()


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# sr_model = ninasr_b0(scale=4, pretrained=True).eval()
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel('../dumps/EDSR_x4.pb')
# sr.setModel("edsr",4)
hairsegmentation = HairSegmentation(1024, 1024)

Tr = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

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
    FACE_OVAL = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
    # Left eye indices list
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    # Right eye indices list
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.6,
                            min_tracking_confidence=0.6) as face_mesh:
            img = cv2.imread(path)
            img  = cv2.resize(img,dsize=(1024, 1024))
            # img = sr.upsample(img)
            # img  = cv2.resize(img,dsize=None,fx=4,fy=4)
            # print(img_sr.shape)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # n_tensor = sr_model(to_tensor(Image.fromarray(rgb)).unsqueeze(0)).squeeze(0)
            # to_pil_image(n_tensor).
            # bgr = cv2.cvtColor(np.array(to_pil_image(n_tensor)), cv2.COLOR_BGR2RGB)
            # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            # img_h, img_w = img.shape[:2]
            img_h, img_w = 1024, 1024
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # create mask for hair
            
            masked_img_hair = hairsegmentation(rgb)
            masked_img_hair = np.stack([masked_img_hair, masked_img_hair, masked_img_hair], axis=-1)
            masked_img = np.zeros_like(img)
            mp_drawing.draw_landmarks(
                    image=masked_img,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            th, im_th = cv2.threshold(masked_img_gray, 200, 255, cv2.THRESH_BINARY_INV)
            im_floodfill = im_th.copy()
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, mask, (0,0), (0,0,0))
            # cv2.imwrite('../dumps/mask.png', im_floodfill)
            masked_img_hair[im_floodfill > 0] = 0
            # create mask for the eyes
            extracted_img = extract_masked_regions(img, masked_img_hair) #hair
            x,y,w,h = cv2.boundingRect(mesh_points[LEFT_EYE])
            left_eye = img[y : y+h, x : x+w, :]
            # cv2.imwrite('../dumps/example_mediapipe_lefteye.png', img[y : y+h, x : x+w, :])
            x,y,w,h = cv2.boundingRect(mesh_points[RIGHT_EYE])
            right_eye = img[y : y+h, x : x+w, :]
            # cv2.imwrite('../dumps/example_mediapipe_righteye.png', img[y : y+h, x : x+w, :])
            # cv2.imwrite('../dumps/example_mediapipe_hair.png', extracted_img)
            cv2.fillPoly(masked_img_hair, [mesh_points[LEFT_EYE]], (255,255,255))
            cv2.fillPoly(masked_img_hair, [mesh_points[RIGHT_EYE]], (255,255,255))
            extracted_img_1 = extract_masked_regions(img, masked_img_hair) # eye and hair
            cv2.imwrite('../dumps/example_mediapipe_hair_eye.png', extracted_img_1)
            extracted_img_2 = extract_masked_regions_overlay(img, masked_img_hair) # overlay on image
            cv2.imwrite('../dumps/example_mediapipe_hair_eye_overlay.png', extracted_img_2)
            with torch.no_grad():
                # L_t = to_tensor(Image.fromarray(cv2.resize(left_eye, (128, 128)))).unsqueeze(0)
                # R_t = to_tensor(Image.fromarray(cv2.resize(right_eye, (128, 128)))).unsqueeze(0)
                # H_t = to_tensor(Image.fromarray(cv2.resize(extracted_img, (128, 128)))).unsqueeze(0)
                L_t = Tr(Image.fromarray(cv2.resize(left_eye, (18, 18)))).unsqueeze(0)
                R_t = Tr(Image.fromarray(cv2.resize(right_eye, (18, 18)))).unsqueeze(0)
                H_t = Tr(Image.fromarray(cv2.resize(extracted_img, (128, 128)))).unsqueeze(0)
                # E = torch.cat([mobilenet_v3_rep(L_t), mobilenet_v3_rep(R_t), mobilenet_v3_rep(H_t)], dim=1)
                # E = torch.cat([mobilenet_v3_rep(L_t), mobilenet_v3_rep(R_t), mobilenet_v3_rep(H_t)], dim=1)
                E = torch.cat([resnet_50(L_t).view(1, -1), resnet_50(R_t).view(1, -1), resnet_50(H_t).view(1, -1)], dim=1)
                E = torch.nn.functional.normalize(E, p = 2, dim=1).squeeze(0).numpy()
                return E
            

# if __name__ == "__main__":
#     if path:
#         # detect_haarcascade(path)
#         detect_mediapipe(path)
