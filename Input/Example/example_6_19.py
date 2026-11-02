"""Streamlit webcam capture and face detection."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def main():
    st.title("Face Detection")
    photo = st.camera_input("Capture webcam image")
    if photo is None:
        return
    image = cv2.cvtColor(np.array(Image.open(photo).convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    for x, y, width, height in detector.detectMultiScale(gray, 1.3, 5):
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 0), 2)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected faces")


if __name__ == "__main__":
    main()
