import os
import sys
import cv2
import face_recognition

img_path = "face1.jpg"
if not os.path.exists(img_path):
    img_path = "Codes/Examples/face1.jpg"

image = cv2.imread(img_path)

if image is None:
    print(f"Error: Gambar '{img_path}' tidak ditemukan!")
    sys.exit(1)

# 1. RESIZE GAMBAR: Bantu detektor HOG membaca wajah yang terlalu besar
height, width = image.shape[:2]
if width > 600:
    scale = 600 / width
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

cv2.imshow('photo', image)

# 2. Konversi warna ke RGB (format yang wajib untuk face_recognition)
rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Cari lokasi wajah secara eksplisit dan hitung jumlahnya
face_locations = face_recognition.face_locations(rgb_frame)
print(f"Total wajah terdeteksi: {len(face_locations)}")

# 4. Cari titik landmark berdasarkan lokasi wajah yang ditemukan
face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
count = 0

for face_landmarks in face_landmarks_list:
    for facial_feature in face_landmarks.keys():
        a = face_landmarks[facial_feature]
        
        # Gambar garis hijau antar titik
        for index, item in enumerate(a):
            if index == len(a) - 1:
                break
            cv2.line(image, item, a[index + 1], [0, 255, 0], 2)
            
        # Gambar titik merah pada tiap landmark
        for pt in face_landmarks[facial_feature]:
            image = cv2.circle(image, pt, 2, (0, 0, 255), 2)
            count += 1

print(f"Total titik landmark digambar: {count}")
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()