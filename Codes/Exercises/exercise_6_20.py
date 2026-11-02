# Exercise 6.20
# Modify the previous Python program (Example 6.22), at file_list = ['emotion10.jpeg'],
# so that it has more than one file. Run the program and comment on the results.
import os
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.5
)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Daftar lebih dari satu file gambar
file_list = ['emotion10.jpeg', 'face0.jpg', 'face1.jpeg', 'face2.jpeg']

print(f"Memproses {len(file_list)} file gambar dengan MediaPipe FaceMesh...")

for idx, file in enumerate(file_list):
    if not os.path.exists(file):
        print(f"File {file} tidak ditemukan, lewati.")
        continue

    image = cv2.imread(file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print(f"[{file}] Tidak ada landmark wajah yang terdeteksi.")
        continue

    annotated_image = image.copy()
    print(f"[{file}] Berhasil mendeteksi {len(results.multi_face_landmarks)} wajah.")

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

    out_name = f'annotated_image_{idx}_{os.path.splitext(file)[0]}.png'
    cv2.imwrite(out_name, annotated_image)
    cv2.imshow(f"Exercise 6.20 - {file}", annotated_image)
    cv2.waitKey(0)

face_mesh.close()
cv2.destroyAllWindows()
