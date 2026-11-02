"""Predict a face label from a trained OpenCV recognizer."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="trained_faces.yml")
    args = parser.parse_args()
    if not hasattr(cv2, "face"):
        raise SystemExit("Install opencv-contrib-python to use cv2.face.")
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    for x, y, width, height in faces:
        label, confidence = recognizer.predict(gray[y:y + height, x:x + width])
        print(f"Label: {label}; confidence: {confidence:.1f}")
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.0f})", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Example 6.11", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
