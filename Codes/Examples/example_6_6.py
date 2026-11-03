# Example 6.6 Face detection by face_recognition library
from PIL import Image
import face_recognition

image = face_recognition.load_image_file("faces.png")
face_locations = face_recognition.face_locations(image)

count = 0
for face_location in face_locations:
    count = count + 1
    top, right, bottom, left = face_location
    print("Face {} Top: {}, Left: {}, Bottom: {}, Right: {}".format(count, top, left, bottom, right))
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
