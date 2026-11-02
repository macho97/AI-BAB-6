# Example 6.4 OpenCV Face, Eye and Smile detection
import cv2

# Membaca gambar
img = cv2.imread('face2.jpeg')

# Memuat classifier untuk wajah, mata, dan senyuman
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

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
