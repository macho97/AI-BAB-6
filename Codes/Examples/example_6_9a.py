# Example 6.9a Face Recognition by face_recognition library using Directory
import face_recognition
import cv2
import numpy as np
import os
import sys

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

testpath = "face2.jpeg"
frame = cv2.imread(testpath)

# Pengecekan agar cv2 tidak memproses variabel kosong jika gambar hilang
if frame is None:
    print(f"Error: Gambar pengujian '{testpath}' tidak ditemukan!")
    sys.exit(1)

# FIX SEGFAULT: Gunakan cvtColor alih-alih [:, :, ::-1]
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
cv2.waitKey(0)
cv2.destroyAllWindows()