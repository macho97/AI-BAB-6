# Example 6.2 OpenCV Face Detection with Webcam
import cv2

# Memuat Haar Cascade classifier untuk deteksi wajah frontal
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Membuka webcam (kamera index 0)
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    # Konversi frame ke grayscale
    gray_img = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray_img, minNeighbors=5)

    # Menggambar kotak dan teks untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(cam_frame, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows()
