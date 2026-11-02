"""Runnable OpenCV/face_recognition examples from Chapter 6.

Examples 6.1-6.8 are selected with --example. Provide image/video paths as
arguments; webcam examples use camera 0 unless --source is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def cv2_module():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install OpenCV first: python -m pip install opencv-python") from exc
    return cv2


def detect_image(image_path: str, features: str = "face") -> None:
    cv2 = cv2_module()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_files = {
        "face": "haarcascade_frontalface_default.xml",
        "eye": "haarcascade_eye.xml",
        "smile": "haarcascade_smile.xml",
    }
    cascades = {}
    for feature in (["face", "eye", "smile"] if features == "all" else [features]):
        cascade_path = Path(cv2.data.haarcascades) / cascade_files[feature]
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load cascade: {cascade_path}")
        cascades[feature] = cascade

    face_cascade = cascades.get("face") or cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / cascade_files["face"])
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.putText(image, "face", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255), 2)
        face_gray = gray[y:y + height, x:x + width]
        face_color = image[y:y + height, x:x + width]
        for feature in ("eye", "smile"):
            if feature not in cascades:
                continue
            found = cascades[feature].detectMultiScale(
                face_gray,
                scaleFactor=1.1 if feature == "eye" else 1.5,
                minNeighbors=10 if feature == "eye" else 5,
                minSize=(10, 10),
            )
            color = (0, 255, 0) if feature == "eye" else (0, 0, 255)
            for fx, fy, fw, fh in found:
                cv2.rectangle(face_color, (fx, fy), (fx + fw, fy + fh), color, 2)

    print(f"Detected {len(faces)} face(s) in {image_path}.")
    cv2.imshow("Chapter 6 example", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(source: str, features: str = "face") -> None:
    cv2 = cv2_module()
    if source.isdigit():
        source = int(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Video source could not be opened: {source}")
    eye = smile = None
    if features == "all":
        eye = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
        smile = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_smile.xml")
        )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(str(cascade_path))
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, width, height in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                if eye is not None and smile is not None:
                    face_gray = gray[y:y + height, x:x + width]
                    face_color = frame[y:y + height, x:x + width]
                    for feature_detector, color, neighbors in (
                        (eye, (0, 255, 0), 8), (smile, (0, 0, 255), 5)
                    ):
                        for fx, fy, fw, fh in feature_detector.detectMultiScale(
                            face_gray, 1.1, neighbors, minSize=(10, 10)
                        ):
                            cv2.rectangle(
                                face_color, (fx, fy), (fx + fw, fy + fh), color, 2
                            )
            cv2.imshow("Chapter 6 video example", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def face_landmarks(image_path: str) -> None:
    cv2 = cv2_module()
    try:
        import face_recognition
    except ImportError as exc:
        raise SystemExit(
            "Install face_recognition and its dependencies before running this example."
        ) from exc
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb_image)
    for face_number, landmarks in enumerate(landmarks_list, start=1):
        for name, points in landmarks.items():
            print(f"Face {face_number} - {name}: {points}")
            for index, point in enumerate(points):
                cv2.circle(image, point, 2, (0, 0, 255), -1)
                if index + 1 < len(points):
                    cv2.line(image, point, points[index + 1], (0, 255, 0), 1)
    print(f"Found landmarks for {len(landmarks_list)} face(s).")
    cv2.imshow("Face landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face_locations(image_paths: list[str]) -> None:
    cv2, face_recognition = cv2_module(), _face_recognition()
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        print(f"{path}: {len(locations)} face(s)")
        for number, (top, right, bottom, left) in enumerate(locations, 1):
            print(f"Face {number}: top={top}, left={left}, bottom={bottom}, right={right}")
            crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2BGR)
            cv2.imshow(f"{Path(path).name} - face {number}", crop)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def _face_recognition():
    try:
        import face_recognition
    except ImportError as exc:
        raise SystemExit("Install face_recognition before running this example.") from exc
    return face_recognition


def landmarks_webcam(camera_index: int) -> None:
    cv2, face_recognition = cv2_module(), _face_recognition()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for landmarks in face_recognition.face_landmarks(rgb):
                for points in landmarks.values():
                    for index, point in enumerate(points):
                        cv2.circle(frame, point, 2, (0, 0, 255), -1)
                        if index + 1 < len(points):
                            cv2.line(frame, point, points[index + 1], (0, 255, 0), 2)
            cv2.imshow("Example 6.8 - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def recognize_webcam(known_dir: str, camera_index: int) -> None:
    cv2, face_recognition = cv2_module(), _face_recognition()
    names, encodings = [], []
    for path in sorted(Path(known_dir).glob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image = face_recognition.load_image_file(str(path))
        found = face_recognition.face_encodings(image)
        if found:
            names.append(path.stem)
            encodings.append(found[0])
        if len(encodings) == 3:
            break
    if not encodings:
        raise ValueError(f"No usable face images were found in {known_dir}")
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, locations)
            for (top, right, bottom, left), encoding in zip(locations, face_encodings):
                distances = face_recognition.face_distance(encodings, encoding)
                best = int(distances.argmin())
                name = names[best] if distances[best] < 0.6 else "Unknown"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 5, max(top - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Example 6.9 - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=(
        "6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9"
    ))
    parser.add_argument("--image", help="Path to an image")
    parser.add_argument("--images", nargs="+", help="Image paths, for example 6.6")
    parser.add_argument("--known-dir", default="known_faces")
    parser.add_argument("--source", default="0", help="Video filename or webcam index")
    parser.add_argument("--features", choices=("face", "eye", "smile", "all"), default="face")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.example in ("6.1", "6.3", "6.4"):
        if not args.image:
            parser.error("--image is required for image examples")
        feature_set = {"6.3": "eye", "6.4": "all"}.get(args.example, args.features)
        detect_image(args.image, feature_set)
    elif args.example in ("6.2", "6.5"):
        detect_video(args.source, "all" if args.example == "6.5" else "face")
    elif args.example == "6.6":
        if not args.images:
            parser.error("--images is required for example 6.6")
        face_locations(args.images)
    elif args.example == "6.7":
        if not args.image:
            parser.error("--image is required for example 6.7")
        face_landmarks(args.image)
    elif args.example == "6.8":
        landmarks_webcam(args.camera)
    elif args.example == "6.9":
        recognize_webcam(args.known_dir, args.camera)
    else:
        parser.error("Choose an example with --example")


if __name__ == "__main__":
    main()
