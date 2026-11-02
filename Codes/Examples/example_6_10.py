# Example 6.10 Train a Face Recognition model with OpenCV
# pip install opencv-contrib-python
import os
import cv2
import numpy as np

# Membuat recognizer LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
imagePaths = ["face0.jpeg", "face1.jpeg", "face2.jpeg"]

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
X = []
ids = []
count = 0

for imagePath in imagePaths:
    if not os.path.exists(imagePath):
        continue
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_numpy = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(
        img_numpy,
        scaleFactor=1.3,
        minNeighbors=10,
        minSize=(100, 100)
    )
    for (x, y, w, h) in faces:
        X.append(img_numpy[y:y + h, x:x + w])
        ids.append(count)
        print(ids)
        break
    count = count + 1

if len(X) > 0 and len(ids) > 0:
    recognizer.train(X, np.array(ids))
    recognizer.save('cvtraining.yml')
    print("Model successfully trained and saved as cvtraining.yml")
else:
    print("No faces detected to train.")
