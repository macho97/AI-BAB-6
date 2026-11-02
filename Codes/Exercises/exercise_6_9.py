# Exercise 6.9
# Modify the previous Python program so that it has 10 known face files. Run the
# program and comment on the accuracy and speed of face recognition on the webcam
# live images.
import os
import time
import face_recognition
import cv2
import numpy as np

# Daftar 10 file gambar wajah yang dikenal
known_face_files = [f"face{i}.jpeg" for i in range(10)]
known_face_encodings = []
known_face_names = []

print("Memuat 10 gambar wajah yang dikenal...")
for f in known_face_files:
    if os.path.exists(f):
        img = face_recognition.load_image_file(f)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(f)[0]
            known_face_names.append(name)
            print(f"  Berhasil memuat: {f} sebagai {name}")
    else:
        print(f"  Peringatan: File {f} tidak ditemukan.")

print(f"Total wajah terdaftar: {len(known_face_names)}")

video_capture = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Menghitung FPS untuk mengukur kecepatan pemrosesan
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        if len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Tampilkan info FPS pada frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Exercise 6.9 - Face Recognition (10 Known Faces)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
