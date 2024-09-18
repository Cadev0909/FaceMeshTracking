import cv2
import time
import mediapipe as mp

#total of 468 points measured on face

class FaceMeshModule():
    def __init__(self, staticMode = False, maxFaceCount = 2, refined_landmarks=False, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5):
        self.staticMode = staticMode
        self.maxFaceCount = maxFaceCount
        self.refined_landmarks = refined_landmarks
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaceCount, self.refined_landmarks,
                                                 self.minDetectionConfidence, self.minTrackingConfidence)  # ctrl click FaceMesh() to see perameters
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLMKS in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLMKS, self.mpFaceMesh.FACEMESH_TESSELATION,
                                      self.drawSpec, self.drawSpec)
                face = []
                for id,lm in enumerate(faceLMKS.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), lm.z
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1) #print id value on face
                    #print(id,x,y,z)
                    face.append([x,y,z])
                faces.append(face)
        return img, faces



def main():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('VidoeDemos/.mp4')
    pTime = 0
    detector = FaceMeshModule()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        # img = detector.findFaceMesh(img, False) #print without drawing

        if len(faces) != 0:
            #print(len(faces)) #print faces it recognizes
            print(faces[0]) #print each point

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        cv2.imshow('Face Mesh', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()