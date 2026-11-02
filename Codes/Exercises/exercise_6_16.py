# Exercise 6.16
# Modify the previous Python program (Example 6.17), choosing a different file
# template for the face swap. Comment on the results.
import os
import sys
import cv2
import dlib
import numpy as np

# Tambahkan path Examples agar FaceSwap_Utils dapat diimpor jika berada di Examples
examples_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Examples"))
if examples_path not in sys.path:
    sys.path.insert(0, examples_path)

try:
    from FaceSwap_Utils import (
        get_image_size,
        get_face_landmarks,
        get_face_mask,
        get_affine_image,
        get_mask_center_point,
        get_mask_union,
        skin_color_adjustment,
    )
except ImportError:
    print("Modul FaceSwap_Utils tidak ditemukan di direktori Examples.")

here = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(here, 'shape_predictor_68_face_landmarks.dat')
# Template wajah baru (misal face2.jpeg alih-alih face1.jpeg)
image_face_path = os.path.join(here, 'face2.jpeg')


def main(template_path=image_face_path):
    if not os.path.exists(predictor_path):
        print(f"Predictor {predictor_path} tidak ditemukan.")
        return

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    im1 = cv2.imread(template_path)
    if im1 is None:
        print(f"File template {template_path} tidak ditemukan.")
        return

    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    landmarks1 = get_face_landmarks(im1, detector, predictor)
    if landmarks1 is None:
        print("Wajah pada template tidak terdeteksi.")
        return

    im1_size = get_image_size(im1)
    im1_mask = get_face_mask(im1_size, landmarks1)
    cam = cv2.VideoCapture(0)

    print("Tekan ESC untuk keluar dari webcam...")
    while True:
        ret_val, im2 = cam.read()
        if not ret_val:
            break
        landmarks2 = get_face_landmarks(im2, detector, predictor)
        if landmarks2 is not None:
            im2_size = get_image_size(im2)
            im2_mask = get_face_mask(im2_size, landmarks2)
            affine_im1 = get_affine_image(im1, im2, landmarks1, landmarks2)
            affine_im1_mask = get_affine_image(im1_mask, im2, landmarks1, landmarks2)
            union_mask = get_mask_union(im2_mask, affine_im1_mask)
            affine_im1 = skin_color_adjustment(affine_im1, im2, mask=union_mask)
            point = get_mask_center_point(affine_im1_mask)
            seamless_im = cv2.seamlessClone(affine_im1, im2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)
            cv2.imshow('Exercise 6.16 - Face Swap', seamless_im)
        else:
            cv2.imshow('Exercise 6.16 - Face Swap', im2)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
