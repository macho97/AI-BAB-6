# Exercise 6.10
# Modify the previous Python program, trying the different OpenCV face recognizers,
# such as EigenFaceRecognizer, FisherFaceRecognizer, and LBPHFaceRecognizer,
# and comment on the accuracy.
import os
import cv2
import numpy as np

# Pilihan tipe recognizer yang tersedia pada modul opencv-contrib-python:
# 1. cv2.face.LBPHFaceRecognizer_create()
# 2. cv2.face.EigenFaceRecognizer_create()
# 3. cv2.face.FisherFaceRecognizer_create()

recognizers = {
    "LBPH": cv2.face.LBPHFaceRecognizer_create(),
    "EigenFace": cv2.face.EigenFaceRecognizer_create(),
    "FisherFace": cv2.face.FisherFaceRecognizer_create(),
}

imagePaths = ["face0.jpeg", "face1.jpeg", "face2.jpeg"]
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Menyiapkan data training dengan ukuran tetap (wajib sama untuk Eigen & Fisher)
TARGET_SIZE = (100, 100)
X = []
ids = []

for idx, imagePath in enumerate(imagePaths):
    if not os.path.exists(imagePath):
        continue
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y + h, x:x + w], TARGET_SIZE)
        X.append(face_roi)
        ids.append(idx)
        break

print(f"Total sampel training: {len(X)} wajah dari {len(set(ids))} kelas.")

testpath = "face2.jpeg"
if os.path.exists(testpath) and len(X) >= 2:
    test_img = cv2.imread(testpath)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_faces = faceCascade.detectMultiScale(test_gray, 1.3, 5, minSize=(50, 50))

    if len(test_faces) > 0:
        (tx, ty, tw, th) = test_faces[0]
        test_roi = cv2.resize(test_gray[ty:ty + th, tx:tx + tw], TARGET_SIZE)

        # Evaluasi ketiga jenis recognizer
        print("\n--- Perbandingan Hasil Prediksi Recognizer ---")
        for name, rec in recognizers.items():
            try:
                rec.train(X, np.array(ids))
                pred_id, conf = rec.predict(test_roi)
                print(f"[{name}] -> Predicted ID: {pred_id}, Confidence (error metric): {conf:.2f}")
            except Exception as e:
                print(f"[{name}] -> Gagal evaluasi: {e}")
else:
    print("Memerlukan minimal 2 sampel wajah berbeda untuk melatih recognizer.")
