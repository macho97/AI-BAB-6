# Exercise 6.4
# Modify the Example 6.4 Python program, trying different images and commenting on
# the accuracy of the detections.
import cv2

test_images = ['face2.jpeg', 'face1.jpeg', 'face0.jpg']
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

for img_name in test_images:
    img = cv2.imread(img_name)
    if img is None:
        print(f"Gambar {img_name} tidak ditemukan, lewati.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"\nGambar: {img_name} | Wajah terdeteksi: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        face_gray = gray[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(10, 10),
        )
        print(f"  - Mata terdeteksi: {len(eyes)}")
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(10, 10),
        )
        print(f"  - Senyuman terdeteksi: {len(smile)}")
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(face_color, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2)

    cv2.imshow(f'Exercise 6.4 - {img_name}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
