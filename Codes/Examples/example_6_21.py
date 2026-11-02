# Example 6.21 Pencil Sketch web app with Streamlit
import streamlit as st
import numpy as np
from PIL import Image
import cv2


def pencilSketch(inp_img):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = cv2.divide(img_gray, 255 - img_smoothing, scale=256)
    return final_img


def main():
    st.title("Photo Pencil Sketch")
    st.write("Convert your photos to pencil sketches")
    file_image = st.sidebar.file_uploader("Upload your Photos", type=['jpeg', 'jpg', 'png', 'gif'])

    if file_image is None:
        st.write("No image file!")
    else:
        input_image = Image.open(file_image)
        final_sketch = pencilSketch(np.array(input_image))
        st.write("Original Photo")
        st.image(input_image, use_column_width=True)
        st.write("Pencil Sketch")
        st.image(final_sketch, use_column_width=True)
        if st.button("Download"):
            im_pil = Image.fromarray(final_sketch)
            im_pil.save('output.jpg')
            st.write('Download completed')


if __name__ == '__main__':
    main()
