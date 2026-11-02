"""Streamlit webcam beautification demo with adjustable image controls."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def main():
    st.title("Webcam Beauty App")
    photo = st.camera_input("Capture webcam image")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)
    smooth = st.sidebar.slider("Smoothness", 1, 25, 5, 2)
    if photo is None:
        return
    image = np.array(Image.open(photo).convert("RGB"))
    adjusted = cv2.convertScaleAbs(image, alpha=1 + contrast / 100, beta=brightness)
    adjusted = cv2.bilateralFilter(adjusted, smooth, 75, 75)
    st.image(np.concatenate((image, adjusted), axis=0), caption="Original and adjusted")


if __name__ == "__main__":
    main()
