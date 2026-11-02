# Example 6.7 Face Landmarks detection by face_recognition library
import cv2
import face_recognition

image = cv2.imread("face1.jpeg")
cv2.imshow('photo', image)

rgb_frame = image[:, :, ::-1]
face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
count = 0

for face_landmarks in face_landmarks_list:
    for facial_feature in face_landmarks.keys():
        print(facial_feature)
        a = face_landmarks[facial_feature]
        for index, item in enumerate(a):
            if index == len(a) - 1:
                break
            cv2.line(image, item, a[index + 1], [0, 255, 0], 2)
        for pt in face_landmarks[facial_feature]:
            image = cv2.circle(image, pt, 2, (0, 0, 255), 2)
            count = count + 1
            print(pt)

print(count)
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
