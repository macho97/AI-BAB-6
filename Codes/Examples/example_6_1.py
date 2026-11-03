# Example 6.1 OpenCV face detection
import cv2

# Membaca gambar wajah
img = cv2.imread('face0.jpg')

# # Memuat Haar Cascade classifier untuk deteksi wajah frontal
# cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Menjadi path built-in OpenCV:
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Konversi gambar ke skala abu-abu (grayscale)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah menggunakan classifier
faces = cascade_classifier.detectMultiScale(gray, minNeighbors=5)

# Menggambar kotak pembatas dan label pada setiap wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(img, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

# Menampilkan hasil gambar
cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
