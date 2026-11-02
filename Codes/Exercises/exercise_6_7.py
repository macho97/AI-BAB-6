# Exercise 6.7
# Modify the previous Python program so that it uses the print() function to print out
# the face landmark names and the corresponding coordinate values.
import cv2
import face_recognition

image = cv2.imread("face1.jpeg")
if image is None:
    print("Gambar face1.jpeg tidak ditemukan.")
    exit(0)

cv2.imshow('photo', image)

rgb_frame = image[:, :, ::-1]
face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

print(f"Jumlah wajah ditemukan: {len(face_landmarks_list)}")

for face_idx, face_landmarks in enumerate(face_landmarks_list, 1):
    print(f"\n=== Landmark untuk Wajah #{face_idx} ===")
    for landmark_name, points in face_landmarks.items():
        print(f"\nFitur: {landmark_name} ({len(points)} titik):")
        for pt_idx, pt in enumerate(points):
            # Mencetak nama landmark dan nilai koordinat (x, y)
            print(f"  {landmark_name}[{pt_idx}] -> Koordinat: X={pt[0]}, Y={pt[1]}")
            cv2.circle(image, pt, 2, (0, 0, 255), 2)

        # Gambar garis penhubung antar titik fitur
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], (0, 255, 0), 2)

cv2.imshow('Exercise 6.7 - Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
