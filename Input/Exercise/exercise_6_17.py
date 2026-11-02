"""Streamlit image upload with face detection and a Save Face button."""

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def main():
    st.title("Exercise 6.17 - Save Face")
    uploaded = st.file_uploader("Choose a photo", type=("jpg", "jpeg", "png"))
    if uploaded is None:
        st.info("Upload a photo to detect faces.")
        return

    image = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Could not load the frontal-face Haar cascade.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 0), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{len(faces)} face(s) detected")
    if st.button("Save Face"):
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("Could not encode the detected image.")
        output = BytesIO(encoded.tobytes())
        st.download_button(
            "Download detected image",
            data=output.getvalue(),
            file_name="face_detected.jpg",
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
