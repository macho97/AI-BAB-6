"""Predict age, gender, ethnicity, and emotion using supplied Keras models."""

import argparse
from pathlib import Path

import cv2
import numpy as np

LABELS = {
    "age": ("below_20", "20-30", "30-40", "40-50", "above_50"),
    "gender": ("Male", "Female"),
    "ethnicity": ("arab", "asian", "black", "hispanic", "indian", "white"),
    "emotion": ("angry", "happy", "neutral", "sad"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--output", default="output.jpg")
    args = parser.parse_args()
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise SystemExit("Install tensorflow before running this example.") from exc
    root = Path(args.model_dir)
    models = {
        key: load_model(str(root / f"{key}_model.h5"))
        for key in ("age", "gender", "ethnicity", "emotion")
    }
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for x, y, width, height in detector.detectMultiScale(gray, 1.3, 5):
        face = cv2.resize(image[y:y + height, x:x + width], (70, 70))
        batch = np.expand_dims(face.astype("float32") / 255.0, axis=0)
        labels = []
        for key, model in models.items():
            prediction = model.predict(batch, verbose=0)[0].argmax()
            labels.append(f"{key}: {LABELS[key][prediction]}")
        for line, label in enumerate(labels):
            cv2.putText(image, label, (x, y + height + 20 + 20 * line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    if not cv2.imwrite(args.output, image):
        raise OSError(f"Could not save result to {args.output}")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
