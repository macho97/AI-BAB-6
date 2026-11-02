"""Streamlit pencil-sketch app with adjustable Gaussian blur parameters."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def pencil_sketch(image, kernel_size, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    smoothed = cv2.GaussianBlur(
        inverted, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma
    )
    return cv2.divide(gray, 255 - smoothed, scale=256)


def main():
    st.title("Exercise 6.19 - Pencil Sketch")
    uploaded = st.file_uploader("Choose a photo", type=("jpg", "jpeg", "png"))
    kernel = st.sidebar.slider("Gaussian kernel (odd)", 3, 51, 21, 2)
    sigma = st.sidebar.slider("Sigma X / Y", 0.0, 20.0, 0.0, 0.5)
    if uploaded is None:
        st.info("Upload a photo to create a sketch.")
        return
    image = np.array(Image.open(uploaded).convert("RGB"))
    result = pencil_sketch(image, kernel, sigma)
    st.image(image, caption="Original")
    st.image(result, caption=f"Sketch (kernel={kernel}, sigma={sigma})")
    st.caption(
        "A larger blur kernel generally removes fine detail and creates smoother "
        "shading; smaller kernels retain more texture."
    )


if __name__ == "__main__":
    main()
