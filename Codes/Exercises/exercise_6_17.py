# Exercise 6.17
# Modify the previous Python program (Example 6.18), adding another Save Face button that,
# when clicked, will save the face-detected image into a file.
import streamlit as st
import cv2
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, faces


def main():
    st.title("Exercise 6.17 - Face Detection with Save Face")
    st.text("Built with Streamlit and OpenCV")

    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'gif'])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

        if st.button("Detect Faces"):
            result_img, result_faces = detect_faces(our_image)
            st.session_state['result_img'] = result_img
            st.session_state['result_faces'] = result_faces

        if 'result_img' in st.session_state:
            result_img = st.session_state['result_img']
            result_faces = st.session_state['result_faces']
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            st.success("Found {} faces".format(len(result_faces)))

            # Tombol untuk menyimpan gambar berwajah yang telah dideteksi ke file
            if st.button("Save Face"):
                save_filename = "saved_detected_face.jpg"
                cv2.imwrite(save_filename, result_img)
                st.success(f"Gambar wajah berhasil disimpan ke {save_filename}!")


if __name__ == '__main__':
    main()
