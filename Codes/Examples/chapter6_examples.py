"""Chapter 6 - Face Detection and Face Recognition: Examples Runner.

This module provides an interactive and CLI menu to run any example from Chapter 6:
  - 6.1:   OpenCV face detection on image (face.py)
  - 6.2:   OpenCV face detection with webcam (facewebcam.py)
  - 6.3:   OpenCV face and eye detection (faceeye.py)
  - 6.4:   OpenCV face, eye and smile detection (faceeyesmile.py)
  - 6.5:   OpenCV face, eye and smile detection with webcam (faceeyesmilewebcam.py)
  - 6.6:   Face detection with face_recognition library (facedetect.py)
  - 6.7:   Face landmarks detection with face_recognition (facemarks.py)
  - 6.8:   Face landmarks detection with webcam (facemarkswebcam.py)
  - 6.9:   Face recognition with webcam (facerecwebcam.py)
  - 6.9a:  Face recognition from directory (facerec2.py)
  - 6.9b:  Face recognition with webcam from directory (facerecwebcam2.py)
  - 6.10:  Train OpenCV face recognizer (facereccv.py)
  - 6.11:  Recognize face with trained OpenCV model (facereccv2.py)
  - 6.12:  Webcam face recognition with trained OpenCV model (facereccvwebcam.py)
  - 6.13:  GUI Face Recognition System with OpenCV (FaceRecSystem1.py)
  - 6.14:  GUI Face Recognition System with face_recognition (FaceRecSystem2.py)
  - 6.15:  DeepFace demo (deepfacedemo.py)
  - 6.15a: DeepFace webcam stream (deepfacestream.py)
  - 6.15b: Demographic predictions with Keras models (get_predictions0.py)
  - 6.16:  Face Swap with OpenCV & face_recognition (faceswap.py)
  - 6.17:  Face Swap with dlib & OpenCV (FaceSwap2.py)
  - 6.18:  Streamlit face detection web app (StreamlitFace.py)
  - 6.19:  Streamlit webcam face detection web app (StreamlitWebFace.py)
  - 6.20:  Streamlit webcam beauty web app (StreamlitWebcamBeauty.py)
  - 6.21:  Streamlit photo pencil sketch web app (pencilsketch_webapp.py)
  - 6.22:  MediaPipe FaceMesh on static image (mediapipe_facemesh.py)
  - 6.23:  MediaPipe FaceMesh on webcam (mediapipe_facemesh_webcam.py)
"""

import argparse
import importlib
import sys


EXAMPLE_MAP = {
    "6.1": "example_6_1",
    "6.2": "example_6_2",
    "6.3": "example_6_3",
    "6.4": "example_6_4",
    "6.5": "example_6_5",
    "6.6": "example_6_6",
    "6.7": "example_6_7",
    "6.8": "example_6_8",
    "6.9": "example_6_9",
    "6.9a": "example_6_9a",
    "6.9b": "example_6_9b",
    "6.10": "example_6_10",
    "6.11": "example_6_11",
    "6.12": "example_6_12",
    "6.13": "example_6_13",
    "6.14": "example_6_14",
    "6.15": "example_6_15",
    "6.15a": "example_6_15a",
    "6.15b": "example_6_15b",
    "6.16": "example_6_16",
    "6.17": "example_6_17",
    "6.18": "example_6_18",
    "6.19": "example_6_19",
    "6.20": "example_6_20",
    "6.21": "example_6_21",
    "6.22": "example_6_22",
    "6.23": "example_6_23",
}


def run_example(example_id: str):
    module_name = EXAMPLE_MAP.get(example_id)
    if not module_name:
        print(f"Unknown example ID: {example_id}. Available: {list(EXAMPLE_MAP.keys())}")
        return
    print(f"--- Running Example {example_id} ({module_name}.py) ---")
    module = importlib.import_module(module_name)
    if hasattr(module, 'main'):
        module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", choices=list(EXAMPLE_MAP.keys()), help="Choose an example to run")
    args = parser.parse_args()

    if args.example:
        run_example(args.example)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
