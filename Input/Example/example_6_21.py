"""Streamlit pencil-sketch photo app."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def main():
    st.title("Photo Pencil Sketch")
    uploaded = st.file_uploader("Upload photo", type=("jpg", "jpeg", "png"))
    if uploaded is None:
        return
    image = np.array(Image.open(uploaded).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    st.image(image, caption="Original")
    st.image(sketch, caption="Pencil sketch")


if __name__ == "__main__":
    main()
