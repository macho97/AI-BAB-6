# Exercise 6.21
# Modify the previous Python program (Example 6.23) and use the print() function to
# print all the face landmarks and corresponding locations. Run the program and
# comment on the results.
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

print("Memulai webcam FaceMesh... Tekan ESC untuk keluar.")

while True:
    success, image = cap.read()
    if not success:
        continue

    # Konversi BGR ke RGB
    rgb_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = face_mesh.process(rgb_image)

    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        h, w, c = annotated_image.shape
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks, 1):
            print(f"\n--- Wajah #{face_idx} ({len(face_landmarks.landmark)} titik landmark) ---")
            for lm_idx, lm in enumerate(face_landmarks.landmark):
                # Hitung koordinat pixel aktual
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                # Cetak semua landmark dan posisinya ke console
                print(f"Landmark {lm_idx}: x={cx}, y={cy}, z={cz:.4f}")

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    cv2.imshow('Exercise 6.21 - MediaPipe FaceMesh with Printed Landmarks', annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
