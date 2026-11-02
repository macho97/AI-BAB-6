# Example 6.18 Face Detection web app with Streamlit
# pip install --upgrade protobuf
# pip install streamlit
import streamlit as st
import cv2
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, faces


def main():
    st.title("Face Detection")
    st.text("Built with Streamlit and OpenCV")
    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'gif'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
        if st.button("Detect Faces"):
            result_img, result_faces = detect_faces(our_image)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            st.success("Found {} faces".format(len(result_faces)))


if __name__ == '__main__':
    main()
