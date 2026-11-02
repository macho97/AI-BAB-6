# Exercise 6.5
# Modify the previous Python program so that it read images from a video file.
# Hint: Modify cv2.VideoCapture(0) and replace 0 with the video filename.
import cv2

video_filename = 'video.mp4'  # Ganti dengan file video uji
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

camera = cv2.VideoCapture(video_filename)

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        print("Selesai memutar video atau tidak dapat membaca frame.")
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

    cv2.imshow('Exercise 6.5 - Video Face, Eye & Smile Detection', frame)
    key = cv2.waitKey(30)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows()
