# Exercise 6.18
# Modify the previous Python program, adding another Save Face button that, when
# clicked, will save the webcam image into a file.
import streamlit as st
import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def main():
    st.title("Exercise 6.18 - Webcam Face Detection with Save Button")
    st.text("Built with Streamlit and OpenCV")

    scale = st.sidebar.slider('Scale Factor:', 1.0, 2.0, 1.3)
    mn = st.sidebar.slider('minNeighbors:', 5, 20, 5)
    msize = st.sidebar.slider('minSize:', 10, 200, 30)

    # Streamlit camera input untuk mengambil gambar langsung dari webcam
    camera_photo = st.camera_input("Ambil gambar webcam")

    if camera_photo is not None:
        file_bytes = camera_photo.getvalue()
        import numpy as np
        image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=mn,
            minSize=(msize, msize)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(image_np, 'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption=f"Terdeteksi {len(faces)} wajah")

        # Tombol Save Face untuk menyimpan hasil tangkapan webcam
        if st.button("Save Face"):
            output_filename = "saved_webcam_face.jpg"
            cv2.imwrite(output_filename, image_np)
            st.success(f"Gambar webcam berhasil disimpan ke {output_filename}!")


if __name__ == '__main__':
    main()
