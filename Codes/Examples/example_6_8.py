# Example 6.8 Face Landmarks detection by face_recognition library using Webcam
import cv2
import face_recognition

# Inisialisasi webcam
video_capture = cv2.VideoCapture(0)

face_locations = []

while True:
    # Mengambil frame dari video webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            a = face_landmarks[facial_feature]
            for index, item in enumerate(a):
                if index == len(a) - 1:
                    break
                cv2.line(frame, item, a[index + 1], [0, 255, 0], 2)
            for pt in face_landmarks[facial_feature]:
                frame = cv2.circle(frame, pt, 2, (0, 0, 255), 2)

    cv2.imshow('Face Landmarks', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
