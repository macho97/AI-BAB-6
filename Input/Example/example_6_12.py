"""Run an OpenCV LBPH recognizer against a live webcam."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="trained_faces.yml")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not hasattr(cv2, "face"):
        raise SystemExit("Install opencv-contrib-python to use cv2.face.")
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a webcam frame.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, width, height in detector.detectMultiScale(gray, 1.3, 5):
                label, confidence = recognizer.predict(gray[y:y + height, x:x + width])
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.0f})", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Example 6.12 - press ESC to quit", frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
