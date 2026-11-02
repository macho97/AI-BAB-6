# Example 6.17a FaceSwap_Utils for Example 6.17
# Modified from:
# https://raw.githubusercontent.com/Jacen789/simple_faceswap/master/faceswap.py
import os
import cv2
import dlib
import numpy as np


def get_image_size(image):
    """
    :param image: image
    :return: (height, width)
    """
    image_size = (image.shape[0], image.shape[1])
    return image_size


def get_face_landmarks(image, face_detector, shape_predictor):
    """
    Get face landmark, 68 key points
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68 key points
    """
    dets = face_detector(image, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        return None
    shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return face_landmarks


def get_face_mask(image_size, face_landmarks):
    """
    Get Face Mask
    :param image_size: Image Size
    :param face_landmarks: 68 key points
    :return: image_mask, Face Mask
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=255)
    return mask


def get_affine_image(image1, image2, face_landmarks1, face_landmarks2):
    """
    Get Affine Image
    :param image1: Image 1
    :param image2: Image 2
    :param face_landmarks1: Image 1 face landmarks
    :param face_landmarks2: Image 2 face landmarks
    :return: The Affine Image 1
    """
    three_points_index = [18, 8, 25]
    M = cv2.getAffineTransform(
        face_landmarks1[three_points_index].astype(np.float32),
        face_landmarks2[three_points_index].astype(np.float32)
    )
    dsize = (image2.shape[1], image2.shape[0])
    affine_image = cv2.warpAffine(image1, M, dsize)
    return affine_image.astype(np.uint8)


def get_mask_center_point(image_mask):
    """
    Get Mask Center Point
    :param image_mask: Image mask
    :return: center point
    """
    image_mask_index = np.argwhere(image_mask > 0)
    if len(image_mask_index) == 0:
        return (0, 0)
    miny, minx = np.min(image_mask_index, axis=0)
    maxy, maxx = np.max(image_mask_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point


def get_mask_union(mask1, mask2):
    """
    Get Mask Union
    :param mask1: mask_image, Mask 1
    :param mask2: mask_image, Mask 2
    :return: The union of two masks
    """
    mask = np.min([mask1, mask2], axis=0)
    mask = ((cv2.blur(mask, (5, 5)) == 255) * 255).astype(np.uint8)
    mask = cv2.blur(mask, (3, 3)).astype(np.uint8)
    return mask


def skin_color_adjustment(im1, im2, mask=None):
    """
    Skin Color Adjustment
    :param im1: Image 1
    :param im2: Image 2
    :param mask: Face mask.
    :return: Adjusted Image 1
    """
    if mask is None:
        im1_ksize = 55
        im2_ksize = 55
        im1_factor = cv2.GaussianBlur(im1, (im1_ksize, im1_ksize), 0).astype(float)
        im2_factor = cv2.GaussianBlur(im2, (im2_ksize, im2_ksize), 0).astype(float)
    else:
        im1_face_image = cv2.bitwise_and(im1, im1, mask=mask)
        im2_face_image = cv2.bitwise_and(im2, im2, mask=mask)
        im1_factor = np.mean(im1_face_image, axis=(0, 1))
        im2_factor = np.mean(im2_face_image, axis=(0, 1))
    im1 = np.clip((im1.astype(float) * im2_factor / np.clip(im1_factor, 1e-6, None)), 0, 255).astype(np.uint8)
    return im1

