# Example 6.20 Webcam Beautify web app with Streamlit
# pip install --upgrade protobuf
# pip install streamlit
from PIL import Image
import streamlit as st
import cv2
import numpy as np


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255.0
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def main():
    st.subheader("Webcam Beauty App")
    st.sidebar.markdown("# Brightness Contrast")
    brightness = st.sidebar.slider("Brightness", 0, 255, 0, 1)
    contrast = st.sidebar.slider("Contrast", 0, 255, 0, 1)
    smothness = st.sidebar.slider("Smothness", 0.0, 5.0, 1.0, 0.1)

    st.sidebar.markdown("# Alpha * frame + Beta")
    alpha = st.sidebar.slider("Alpha", 0.0, 5.0, 2.0, 0.1)
    beta = st.sidebar.slider("Beta", 0, 100, 10, 1)

    cap = cv2.VideoCapture(0)
    frameST = st.empty()
    stop = st.button("Stop")

    while not stop:
        ret, frame = cap.read()
        if not ret:
            break

        # Alpha and Beta
        frame2 = np.uint8(np.clip((alpha * frame + beta), 0, 255))

        # Adjust the Smoothness of the Image
        level = int(smothness * 10)
        if level > 0:
            frame2 = cv2.bilateralFilter(frame2, level, 75, 75)

        # Adjust the brightness and contrast
        frame2 = apply_brightness_contrast(frame2, brightness, contrast)

        # Concatenate image Vertically
        result = np.concatenate((frame, frame2), axis=0)
        frameST.image(result, channels="BGR")

    cap.release()


if __name__ == '__main__':
    main()
