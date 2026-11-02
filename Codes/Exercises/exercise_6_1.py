# Exercise 6.1
# Modify the Example 6.1 Python program. Try different face images and comment on
# the accuracy of the face detection. Adjust the scaleFactor, minNeighbors, and
# minSize parameters to see how they affect the results.
import cv2

# Gambar uji (dapat diganti dengan nama file gambar wajah lainnya)
image_path = 'face0.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Gambar {image_path} tidak ditemukan.")
    exit(0)

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Parameter yang dapat disesuaikan:
# - scaleFactor: menentukan seberapa banyak ukuran gambar dikurangi pada setiap skala gambar (misal 1.1 - 1.5)
# - minNeighbors: menentukan berapa banyak tetangga yang harus dimiliki setiap persegi panjang kandidat untuk dipertahankan
# - minSize: ukuran minimum objek yang mungkin (objek yang lebih kecil dari ini diabaikan)
scale_factor = 1.3
min_neighbors = 5
min_size = (30, 30)

faces = cascade_classifier.detectMultiScale(
    gray,
    scaleFactor=scale_factor,
    minNeighbors=min_neighbors,
    minSize=min_size
)

print(f"Parameter: scaleFactor={scale_factor}, minNeighbors={min_neighbors}, minSize={min_size}")
print(f"Jumlah wajah terdeteksi: {len(faces)}")

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(img, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

cv2.imshow('Exercise 6.1 - Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
