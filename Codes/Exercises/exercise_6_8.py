# Exercise 6.8
# Modify the previous Python program so that it uses the cv2.putText() function to
# display the face landmark names at the corresponding positions on the webcam live
# images.
import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        for facial_feature, points in face_landmarks.items():
            # Menggambar titik dan garis landmark
            for index, item in enumerate(points):
                if index < len(points) - 1:
                    cv2.line(frame, item, points[index + 1], [0, 255, 0], 1)
                frame = cv2.circle(frame, item, 2, (0, 0, 255), -1)

            # Menampilkan nama landmark di posisi titik pertama dari fitur tersebut
            if points:
                first_pt = points[0]
                cv2.putText(
                    frame,
                    facial_feature,
                    (first_pt[0] - 10, first_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )

    cv2.imshow('Exercise 6.8 - Face Landmarks with Labels', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
