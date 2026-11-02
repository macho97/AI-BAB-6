"""OpenCV student face system: register samples, train, and recognize."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


FIELDS = ("Id", "Name", "Course", "Address")


def register(args, detector):
    args.data_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    sample = 0
    try:
        while sample < args.samples:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a webcam frame.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            for x, y, width, height in faces:
                crop = cv2.resize(gray[y:y + height, x:x + width], (200, 200))
                output = args.data_dir / f"{args.id}.{sample}.jpg"
                if not cv2.imwrite(str(output), crop):
                    raise OSError(f"Could not save sample {output}")
                sample += 1
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"Saved {sample}/{args.samples}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if sample >= args.samples:
                    break
            cv2.imshow("Register student - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if sample == 0:
        raise RuntimeError("No face samples were captured.")
    new_file = not args.database.exists() or args.database.stat().st_size == 0
    with args.database.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(FIELDS)
        writer.writerow((args.id, args.name, args.course, args.address))
    print(f"Saved {sample} samples and student information.")


def train(args):
    if not hasattr(cv2, "face"):
        raise SystemExit("Install opencv-contrib-python to use cv2.face.")
    images, labels = [], []
    for path in sorted(args.data_dir.glob("*.jpg")):
        try:
            student_id = int(path.stem.split(".", maxsplit=1)[0])
        except ValueError:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(cv2.resize(image, (200, 200)))
            labels.append(student_id)
    if not images:
        raise ValueError(f"No registered face images found in {args.data_dir}")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.asarray(labels, dtype=np.int32))
    model.save(str(args.model))
    print(f"Trained model with {len(images)} samples: {args.model}")


def recognize(args, detector):
    if not hasattr(cv2, "face"):
        raise SystemExit("Install opencv-contrib-python to use cv2.face.")
    if not args.model.is_file() or not args.database.is_file():
        raise FileNotFoundError("Train a model and register student records first.")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(str(args.model))
    with args.database.open(newline="", encoding="utf-8") as file:
        students = {int(row["Id"]): row for row in csv.DictReader(file)}
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
                student_id, confidence = model.predict(
                    cv2.resize(gray[y:y + height, x:x + width], (200, 200))
                )
                student = students.get(student_id)
                label = student["Name"] if student and confidence < args.threshold else "Unknown"
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.0f})", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Recognize student - press ESC to stop", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "train", "recognize"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--database", type=Path, default=Path("database.txt"))
    parser.add_argument("--model", type=Path, default=Path("Trainner.yml"))
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--name", default="")
    parser.add_argument("--course", default="")
    parser.add_argument("--address", default="")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Could not load the frontal-face Haar cascade.")
    if args.mode == "register":
        if not args.name:
            parser.error("--name is required to register a student")
        register(args, detector)
    elif args.mode == "train":
        train(args)
    else:
        recognize(args, detector)


if __name__ == "__main__":
    main()
