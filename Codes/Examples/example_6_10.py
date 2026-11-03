import os
import cv2
import numpy as np

# Membuat recognizer LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
imagePaths = ["face0.jpeg", "face1.jpeg", "face2.jpeg"]

# FIX 1: Gunakan path built-in OpenCV untuk file XML Haar Cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
X = []
ids = []
count = 0

for imagePath in imagePaths:
    # FIX 2: Cek apakah gambar berada di root atau di dalam subfolder
    if not os.path.exists(imagePath):
        imagePath = os.path.join("Codes", "Examples", imagePath)
        if not os.path.exists(imagePath):
            print(f"Melewati {imagePath} (Gambar tidak ditemukan)")
            continue

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_numpy = np.array(gray, 'uint8')
    
    # Deteksi wajah (minNeighbors diturunkan ke 5 agar tidak terlalu kaku saat training)
    faces = faceCascade.detectMultiScale(
        img_numpy,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    for (x, y, w, h) in faces:
        X.append(img_numpy[y:y + h, x:x + w])
        ids.append(count)
        print(f"Wajah diproses: ID {count} dari file {imagePath}")
        break
    count = count + 1

if len(X) > 0 and len(ids) > 0:
    recognizer.train(X, np.array(ids))
    recognizer.save('cvtraining.yml')
    print("Model successfully trained and saved as cvtraining.yml")
else:
    print("No faces detected to train.")