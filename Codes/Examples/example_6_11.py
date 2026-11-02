# Example 6.11 Image Face Recognition with model trained in OpenCV
# pip install opencv-contrib-python
import os
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('cvtraining.yml')

testpath = "face2.jpeg"
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread(testpath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img_numpy = np.array(gray, 'uint8')
faces = faceCascade.detectMultiScale(
    img_numpy,
    scaleFactor=1.3,
    minNeighbors=10,
    minSize=(100, 100)
)

for (x, y, w, h) in faces:
    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
    print(id)
    print(confidence)
