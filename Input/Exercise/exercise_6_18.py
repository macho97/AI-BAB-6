"""Streamlit camera capture with a button to save the webcam image."""

import streamlit as st


def main():
    st.title("Exercise 6.18 - Save Webcam Image")
    photo = st.camera_input("Take a webcam photo")
    if photo is not None and st.button("Save Face"):
        output_path = "webcam_capture.jpg"
        with open(output_path, "wb") as output:
            output.write(photo.getvalue())
        st.success(f"Saved webcam image to {output_path}")
        st.download_button(
            "Download image",
            data=photo.getvalue(),
            file_name=output_path,
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
