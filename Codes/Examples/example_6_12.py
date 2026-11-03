# Example 6.12 Webcam Face Recognition with model trained in OpenCV
import os
import sys
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists('cvtraining.yml'):
    print("Error: cvtraining.yml tidak ditemukan.")
    sys.exit(1)

recognizer.read('cvtraining.yml')

# FIX 1: Gunakan path built-in OpenCV dan varian 'alt2' yang lebih responsif
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FIX 2: Longgarkan parameter agar mudah menangkap wajah bergerak di webcam
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        
        # Tampilkan teks ID dan Confidence di atas & bawah wajah
        cv2.putText(img, f"ID: {id}", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Conf: {int(confidence)}", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Tekan 'ESC' untuk keluar
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()