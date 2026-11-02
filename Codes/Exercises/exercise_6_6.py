# Exercise 6.6
# Modify the previous Python program, trying different images with multiple faces.
# Then run the program and comment on the accuracy.
import cv2
import face_recognition

test_images = ["faces.jpg", "group_photo.jpg"]

for img_path in test_images:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Gambar {img_path} tidak ditemukan, lewati.")
        continue

    print(f"\nMemproses gambar: {img_path}")
    rgb_frame = image[:, :, ::-1]  # Konversi BGR ke RGB untuk face_recognition
    face_locations = face_recognition.face_locations(rgb_frame)

    print(f"Jumlah wajah ditemukan: {len(face_locations)}")

    count = 0
    for face_location in face_locations:
        count = count + 1
        top, right, bottom, left = face_location
        print(f"  Face {count} -> Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
        face_crop = image[top:bottom, left:right]
        cv2.imshow(f"{img_path} - Face {count}", face_crop)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"Face {count}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow(f"Overview - {img_path}", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
