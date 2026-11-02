"""Chapter 6 - Face Detection and Face Recognition: Exercises Runner.

This module provides an interactive and CLI menu to run any exercise from Chapter 6:
  - 6.1:  Parameter adjustment on OpenCV face detection (exercise_6_1.py)
  - 6.2:  Video file face detection with OpenCV (exercise_6_2.py)
  - 6.3:  Face and eye detection on multiple images (exercise_6_3.py)
  - 6.4:  Face, eye, and smile detection accuracy test (exercise_6_4.py)
  - 6.5:  Video face, eye, and smile detection (exercise_6_5.py)
  - 6.6:  Face detection on multiple faces (exercise_6_6.py)
  - 6.7:  Print face landmark names and coordinates (exercise_6_7.py)
  - 6.8:  Webcam face landmarks with labeled text (exercise_6_8.py)
  - 6.9:  Webcam face recognition with 10 known faces & speed test (exercise_6_9.py)
  - 6.10: Compare LBPH, EigenFace, and FisherFace recognizers (exercise_6_10.py)
  - 6.11: Student record system with Address attribute (exercise_6_11.py)
  - 6.12: Student system with custom database filename input (exercise_6_12.py)
  - 6.13: DeepFace demographic analysis on diverse photos (exercise_6_13.py)
  - 6.14: Keras demographic predictions on diverse photos (exercise_6_14.py)
  - 6.15: Face swap testing across different photo pairs (exercise_6_15.py)
  - 6.16: Face swap with different face template (exercise_6_16.py)
  - 6.17: Streamlit face detector with Save Face button (exercise_6_17.py)
  - 6.18: Streamlit webcam detector with Save Face button (exercise_6_18.py)
  - 6.19: Streamlit pencil sketch with blur sliders (exercise_6_19.py)
  - 6.20: MediaPipe FaceMesh on multiple image files (exercise_6_20.py)
  - 6.21: MediaPipe FaceMesh with landmark coordinate printing (exercise_6_21.py)
"""

import argparse
import importlib


EXERCISE_MAP = {
    f"6.{i}": f"exercise_6_{i}" for i in range(1, 22)
}


def run_exercise(exercise_id: str):
    module_name = EXERCISE_MAP.get(exercise_id)
    if not module_name:
        print(f"Unknown exercise ID: {exercise_id}. Available: {list(EXERCISE_MAP.keys())}")
        return
    print(f"--- Running Exercise {exercise_id} ({module_name}.py) ---")
    module = importlib.import_module(module_name)
    if hasattr(module, 'main'):
        module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exercise", nargs="?", choices=list(EXERCISE_MAP.keys()), help="Choose an exercise to run")
    args = parser.parse_args()

    if args.exercise:
        run_exercise(args.exercise)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
