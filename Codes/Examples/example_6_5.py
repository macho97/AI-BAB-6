# Example 6.5 OpenCV Face, Eye and Smile detection with Webcam
import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(10, 10),
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.6,
            minNeighbors=5,
            minSize=(10, 10),
        )
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(face_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)

    cv2.imshow('face', frame)
    key = cv2.waitKey(30)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows()
