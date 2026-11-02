# Example 6.22 mediapipe facemesh
# Modified based on:
# https://google.github.io/mediapipe/solutions/face_mesh.html
# pip install mediapipe
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

file_list = ['emotion10.jpeg']
for idx, file in enumerate(file_list):
    image = cv2.imread(file)
    if image is None:
        print(f"File {file} not found.")
        continue

    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        print('face_landmarks:', face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
    cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)
    cv2.imshow(file, annotated_image)

face_mesh.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
