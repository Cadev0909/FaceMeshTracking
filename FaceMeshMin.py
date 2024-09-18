import cv2
import time
import mediapipe as mp

#total of 468 points measured on face

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('VidoeDemos/.mp4')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) #ctrl click FaceMesh() to see perameters

drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLMKS in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLMKS, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec)
            for id,lm in enumerate(faceLMKS.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x, y, z = int(lm.x * iw), int(lm.y * ih), lm.z
                print(id,x,y,z)

    pTime = 0
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 2)

    cv2.imshow('Face Mesh', img)
    cv2.waitKey(1)