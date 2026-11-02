# Exercise 6.2
# Modify the Example 6.2 Python program so that it reads images from a video file.
# Hint: Modify cv2.VideoCapture(0) and replace 0 with the video filename.
import cv2

video_filename = 'video.mp4'  # Ganti dengan path file video yang diinginkan
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Membuka file video
camera = cv2.VideoCapture(video_filename)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        print("Selesai memutar video atau tidak dapat membaca frame.")
        break

    gray_img = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray_img, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(cam_frame, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Exercise 6.2 - Video Face Detection', cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows()
