# Exercise 6.19
# Modify the previous Python program (Example 6.21), add sliders for adjusting the parameters in
# the cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0) function. Comment on the results.
import streamlit as st
import numpy as np
from PIL import Image
import cv2


def pencilSketch(inp_img, ksize, sigma_x, sigma_y):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    # GaussianBlur dengan kernel dan sigma yang dapat disesuaikan lewat slider
    img_smoothing = cv2.GaussianBlur(img_invert, (ksize, ksize), sigmaX=sigma_x, sigmaY=sigma_y)
    final_img = cv2.divide(img_gray, 255 - img_smoothing, scale=256)
    return final_img


def main():
    st.title("Exercise 6.19 - Photo Pencil Sketch with GaussianBlur Sliders")
    st.write("Sesuaikan parameter Gaussian Blur untuk melihat pengaruh kehalusan dan detail sketsa pensil.")

    st.sidebar.header("Parameter GaussianBlur")
    # Kernel size harus bilangan ganjil positif
    kernel_size = st.sidebar.slider("Kernel Size (Ganjil)", min_value=3, max_value=99, value=21, step=2)
    sigma_x = st.sidebar.slider("Sigma X", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    sigma_y = st.sidebar.slider("Sigma Y", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

    file_image = st.sidebar.file_uploader("Upload your Photos", type=['jpeg', 'jpg', 'png', 'gif'])

    if file_image is None:
        st.write("Silakan unggah file gambar!")
    else:
        input_image = Image.open(file_image)
        final_sketch = pencilSketch(np.array(input_image), kernel_size, sigma_x, sigma_y)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Photo")
            st.image(input_image, use_column_width=True)
        with col2:
            st.write(f"Pencil Sketch (Kernel: {kernel_size}, SigmaX: {sigma_x}, SigmaY: {sigma_y})")
            st.image(final_sketch, use_column_width=True)

        if st.button("Download"):
            im_pil = Image.fromarray(final_sketch)
            im_pil.save('output_sketch.jpg')
            st.write('Download completed as output_sketch.jpg')


if __name__ == '__main__':
    main()
