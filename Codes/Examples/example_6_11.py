import os
import sys
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists('cvtraining.yml'):
    print("Error: cvtraining.yml tidak ditemukan.")
    sys.exit(1)
    
recognizer.read('cvtraining.yml')

testpath = "face2.jpeg"
if not os.path.exists(testpath):
    testpath = os.path.join("Codes", "Examples", "face2.jpeg")

# FIX 1: Gunakan model 'alt2' yang jauh lebih kuat dan akurat dari versi 'default'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
image = cv2.imread(testpath)

if image is None:
    print(f"Error: Gambar '{testpath}' tidak ditemukan!")
    sys.exit(1)

# Perkecil dimensi gambar agar pemindaian lebih ringan dan tidak terlewat
height, width = image.shape[:2]
if width > 800:
    scale = 800 / width
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_numpy = np.array(gray, 'uint8')

# FIX 2: Parameter pemindaian dibuat sangat ketat (scaleFactor kecil) dan toleran (minNeighbors kecil)
faces = faceCascade.detectMultiScale(
    img_numpy,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(30, 30)
)

if len(faces) == 0:
    print("Wajah masih gagal dideteksi. Pastikan face2.jpeg menampilkan wajah dari depan secara jelas.")
else:
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print(id)
        print(confidence)