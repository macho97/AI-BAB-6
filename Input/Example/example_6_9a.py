"""Recognize faces in a still image against JPEG files in a known-faces folder."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-dir", default="known_faces")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    try:
        import face_recognition
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Install face_recognition and numpy before running this example.") from exc
    names, encodings = [], []
    for path in sorted(Path(args.known_dir).glob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image = face_recognition.load_image_file(str(path))
        found = face_recognition.face_encodings(image)
        if found:
            names.append(path.stem)
            encodings.append(found[0])
    if not encodings:
        raise ValueError(f"No usable known face images found in {args.known_dir}")
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings_in_image = face_recognition.face_encodings(rgb, locations)
    for (top, right, bottom, left), encoding in zip(locations, encodings_in_image):
        distances = face_recognition.face_distance(encodings, encoding)
        best = int(np.argmin(distances))
        label = names[best] if distances[best] < 0.6 else "Unknown"
        print(f"{label}: distance={distances[best]:.3f}")
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, label, (left + 5, max(top - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Example 6.9a", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
