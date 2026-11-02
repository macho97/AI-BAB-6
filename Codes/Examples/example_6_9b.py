# Example 6.9b Face Recognition by face_recognition library using Webcam
import face_recognition
import cv2
import numpy as np
import os

path = '.'

def getFaces(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    count = 0
    for imagepath in imagepaths:
        filename = os.path.split(imagepath)[-1]
        parts = filename.split('.')
        if len(parts) < 2 or parts[1] != 'jpeg':
            continue
        print(imagepath)
        face_image = face_recognition.load_image_file(imagepath)
        encodings = face_recognition.face_encodings(face_image)
        if len(encodings) > 0:
            faces.append(encodings[0])
            ID = parts[0]
            IDs.append(ID)
            count = count + 1
    return IDs, faces

known_face_names, known_face_encodings = getFaces(path)
print(known_face_names)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
