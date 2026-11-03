import os
import sys
import cv2

# 1. Gunakan path built-in OpenCV untuk seluruh file XML Haar Cascade
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'

faceCascade = cv2.CascadeClassifier(face_cascade_path)
eyeCascade = cv2.CascadeClassifier(eye_cascade_path)
smileCascade = cv2.CascadeClassifier(smile_cascade_path)

# 2. Cek lokasi gambar (root folder atau subfolder Codes/Examples/)
img_path = 'face2.jpeg'
if not os.path.exists(img_path):
    img_path = 'face2.jpg'

img = cv2.imread(img_path)

# 3. Validasi keberadaan gambar sebelum diproses
if img is None:
    print(f"Error: Gambar '{img_path}' tidak ditemukan! Pastikan file gambar ada di folder project.")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(img, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    face_gray = gray[y:y + h, x:x + w]
    face_color = img[y:y + h, x:x + w]

    # Deteksi mata
    eyes = eyeCascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(10, 10),
    )
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Deteksi senyuman
    smile = smileCascade.detectMultiScale(
        face_gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(10, 10),
    )
    for (xx, yy, ww, hh) in smile:
        cv2.rectangle(face_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)

cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()