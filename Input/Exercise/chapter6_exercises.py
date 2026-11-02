"""Ordered Chapter 6 exercise solutions.

Run a solution with, for example:
    python chapter6_exercises.py 6.1 --image face.jpg

Optional libraries are imported only by the exercise that needs them.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def cv2_module():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install OpenCV first: python -m pip install opencv-python") from exc
    return cv2


def cascade_file(cv2, name: str):
    path = Path(cv2.data.haarcascades) / name
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {path}")
    return detector


def exercise_6_1(args):
    cv2 = cv2_module()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_file(cv2, "haarcascade_frontalface_default.xml").detectMultiScale(
        gray, scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors,
        minSize=(args.min_size, args.min_size),
    )
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    print(f"Detected {len(faces)} face(s). Smaller scaleFactor / minNeighbors often "
          "finds more candidates but can increase false positives.")
    cv2.imshow("Exercise 6.1", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercise_6_2(args):
    _haar_video(args, with_features=False)


def exercise_6_3(args):
    _haar_image(args, smile=False)


def exercise_6_4(args):
    _haar_image(args, smile=True)


def _haar_image(args, smile: bool):
    cv2 = cv2_module()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = cascade_file(cv2, "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    eye_detector = cascade_file(cv2, "haarcascade_eye.xml")
    smile_detector = cascade_file(cv2, "haarcascade_smile.xml") if smile else None
    for x, y, w, h in faces:
        roi_gray, roi_color = gray[y:y + h, x:x + w], image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for ex, ey, ew, eh in eye_detector.detectMultiScale(roi_gray, 1.1, 8, minSize=(10, 10)):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        if smile_detector:
            for sx, sy, sw, sh in smile_detector.detectMultiScale(
                roi_gray, 1.5, 5, minSize=(10, 10)
            ):
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    print(f"Detected {len(faces)} face(s); eye/smile results depend on pose, lighting, "
          "and image quality.")
    cv2.imshow("Exercise 6.4" if smile else "Exercise 6.3", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _haar_video(args, with_features: bool):
    cv2 = cv2_module()
    source = args.video if args.video else "0"
    if source.isdigit():
        source = int(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    face_detector = cascade_file(cv2, "haarcascade_frontalface_default.xml")
    eye_detector = cascade_file(cv2, "haarcascade_eye.xml") if with_features else None
    smile_detector = cascade_file(cv2, "haarcascade_smile.xml") if with_features else None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, w, h in face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30)):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                if with_features:
                    roi_gray, roi_color = gray[y:y + h, x:x + w], frame[y:y + h, x:x + w]
                    for detector, color in ((eye_detector, (0, 255, 0)),
                                            (smile_detector, (0, 0, 255))):
                        for fx, fy, fw, fh in detector.detectMultiScale(roi_gray, 1.1, 5):
                            cv2.rectangle(roi_color, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.imshow("Exercise video", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def exercise_6_5(args):
    _haar_video(args, with_features=True)


def _face_recognition():
    try:
        import face_recognition
    except ImportError as exc:
        raise SystemExit("Install face_recognition before running this exercise.") from exc
    return face_recognition


def exercise_6_6(args):
    cv2, face_recognition = cv2_module(), _face_recognition()
    paths = [Path(args.image)] if args.image else sorted(
        path for path in Path(args.images_dir).iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    total = 0
    for path in paths:
        image = face_recognition.load_image_file(str(path))
        locations = face_recognition.face_locations(image)
        total += len(locations)
        print(f"{path}: {len(locations)} face(s)")
        for top, right, bottom, left in locations:
            cv2.imshow(f"Face in {path.name}", cv2.cvtColor(
                image[top:bottom, left:right], cv2.COLOR_RGB2BGR
            ))
            cv2.waitKey(0)
    print(f"Total detected faces: {total}. Frontal, well-lit faces generally perform better.")
    cv2.destroyAllWindows()


def exercise_6_7(args):
    cv2, face_recognition = cv2_module(), _face_recognition()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    landmarks = face_recognition.face_landmarks(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for index, face in enumerate(landmarks, 1):
        for name, points in face.items():
            print(f"Face {index}, {name}: {points}")
    if not landmarks:
        print("No face landmarks detected.")


def exercise_6_8(args):
    cv2, face_recognition = cv2_module(), _face_recognition()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for face in face_recognition.face_landmarks(rgb):
                for name, points in face.items():
                    if points:
                        x, y = points[0]
                        cv2.putText(frame, name, (x, max(y - 4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        for point in points:
                            cv2.circle(frame, point, 1, (0, 0, 255), -1)
            cv2.imshow("Exercise 6.8 - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def exercise_6_9(args):
    cv2, face_recognition = cv2_module(), _face_recognition()
    known_names, encodings = [], []
    for path in sorted(Path(args.known_dir).glob("*"))[:10]:
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image = face_recognition.load_image_file(str(path))
        found = face_recognition.face_encodings(image)
        if found:
            known_names.append(path.stem)
            encodings.append(found[0])
    if not encodings:
        raise ValueError(f"No usable face images found in {args.known_dir}")
    print(f"Loaded {len(encodings)} of up to 10 known faces.")
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            start = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            detected = face_recognition.face_encodings(rgb, locations)
            for (top, right, bottom, left), encoding in zip(locations, detected):
                distances = face_recognition.face_distance(encodings, encoding)
                best = int(distances.argmin())
                name = known_names[best] if distances[best] < args.tolerance else "Unknown"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 5, max(top - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            fps = 1.0 / max(time.perf_counter() - start, 1e-6)
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("Exercise 6.9 - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print("Recognition quality depends on the known images and lighting; FPS is shown live.")


def exercise_6_10(args):
    cv2 = cv2_module()
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Install numpy before running this exercise.") from exc
    if not hasattr(cv2, "face"):
        raise SystemExit("Install opencv-contrib-python to use cv2.face recognizers.")
    constructors = {
        "eigen": cv2.face.EigenFaceRecognizer_create,
        "fisher": cv2.face.FisherFaceRecognizer_create,
        "lbph": cv2.face.LBPHFaceRecognizer_create,
    }
    image_paths = sorted(Path(args.dataset).glob("*/*"))
    images, labels, label_names = [], [], {}
    for path in image_paths:
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        label = path.parent.name
        label_names.setdefault(label, len(label_names))
        images.append(cv2.resize(gray, (200, 200)))
        labels.append(label_names[label])
    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain at least two person subfolders.")
    model = constructors[args.recognizer]()
    model.train(images, np.array(labels))
    model.save(args.model)
    print(f"Trained {args.recognizer} recognizer on {len(images)} images; saved to {args.model}.")


def exercise_6_11(args):
    fields = ("Id", "Name", "Course", "Address")
    _append_student(args.database, fields, [args.id, args.name, args.course, args.address])


def exercise_6_12(args):
    fields = ("Id", "Name", "Course", "Address")
    _append_student(args.database, fields, [args.id, args.name, args.course, args.address])


def _append_student(database: str, fields, row) -> None:
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(fields)
        writer.writerow(row)
    print(f"Student record saved in {path}")


def exercise_6_13(args):
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise SystemExit("Install deepface before running this exercise.") from exc
    detections = DeepFace.extract_faces(img_path=args.image, enforce_detection=True)
    print(f"Detected {len(detections)} face(s) in {args.image}.")
    comparison = DeepFace.verify(args.image, args.compare_image)
    print(f"Image verification result: {comparison['verified']}")
    analysis = DeepFace.analyze(
        img_path=args.image, actions=["age", "gender", "race", "emotion"]
    )
    print(analysis)
    print("Detection and analysis quality can vary with pose, lighting, and image resolution.")


def exercise_6_14(args):
    try:
        import numpy as np
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise SystemExit("Install numpy and tensorflow to run this exercise.") from exc
    cv2 = cv2_module()
    labels = {
        "age": ("below_20", "20-30", "30-40", "40-50", "above_50"),
        "gender": ("Male", "Female"),
        "ethnicity": ("arab", "asian", "black", "hispanic", "indian", "white"),
        "emotion": ("angry", "happy", "neutral", "sad"),
    }
    models = {
        name: load_model(str(Path(args.model_dir) / f"{name}_model.h5"))
        for name in labels
    }
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    detector = cascade_file(cv2, "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    for x, y, width, height in faces:
        roi = cv2.resize(image[y:y + height, x:x + width], (70, 70))
        batch = np.expand_dims(roi.astype("float32") / 255.0, axis=0)
        details = []
        for name, model in models.items():
            index = int(model.predict(batch, verbose=0)[0].argmax())
            details.append(f"{name}: {labels[name][index]}")
        for line, detail in enumerate(details):
            cv2.putText(image, detail, (x, y + height + 20 * (line + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    if not cv2.imwrite(args.output, image):
        raise OSError(f"Could not save result: {args.output}")
    print(f"Found {len(faces)} face(s). Saved predictions to {args.output}")
    print("Predictions depend on the supplied pretrained models and the input photo.")


def _deepface_analyze(image: str, actions) -> None:
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise SystemExit("Install deepface before running this exercise.") from exc
    result = DeepFace.analyze(img_path=image, actions=list(actions), enforce_detection=True)
    print(result)


def exercise_6_15(args):
    _face_swap(args.source_image, args.target_image, args.output)


def exercise_6_16(args):
    _face_swap(args.source_image, args.target_image, args.output)


def _face_swap(source_path: str, target_path: str, output_path: str) -> None:
    cv2, face_recognition = cv2_module(), _face_recognition()
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    if source is None or target is None:
        raise FileNotFoundError("Could not read the source or target face image.")
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source_faces = face_recognition.face_landmarks(source_rgb)
    target_faces = face_recognition.face_landmarks(target_rgb)
    if not source_faces or not target_faces:
        raise ValueError("Both images must contain a detectable face.")
    source_points = source_faces[0].get("chin", [])
    target_points = target_faces[0].get("chin", [])
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("Face landmarks are incomplete.")
    source_hull = cv2.convexHull(__import__("numpy").array(source_points, dtype="int32"))
    target_hull = cv2.convexHull(__import__("numpy").array(target_points, dtype="int32"))
    mask = __import__("numpy").zeros(target.shape[:2], dtype="uint8")
    cv2.fillConvexPoly(mask, target_hull, 255)
    source_rect = cv2.boundingRect(source_hull)
    target_rect = cv2.boundingRect(target_hull)
    source_crop = source[source_rect[1]:source_rect[1] + source_rect[3],
                         source_rect[0]:source_rect[0] + source_rect[2]]
    warped = cv2.resize(source_crop, (target_rect[2], target_rect[3]))
    warped_full = target.copy()
    warped_full[target_rect[1]:target_rect[1] + target_rect[3],
                target_rect[0]:target_rect[0] + target_rect[2]] = warped
    center = (target_rect[0] + target_rect[2] // 2, target_rect[1] + target_rect[3] // 2)
    result = cv2.seamlessClone(warped_full, target, mask, center, cv2.NORMAL_CLONE)
    if not cv2.imwrite(output_path, result):
        raise OSError(f"Could not write result: {output_path}")
    print(f"Saved face-swap result to {output_path}; alignment quality varies with pose.")


def exercise_6_17(args):
    cv2 = cv2_module()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_file(cv2, "haarcascade_frontalface_default.xml").detectMultiScale(
        gray, 1.3, 5, minSize=(30, 30)
    )
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    if not cv2.imwrite(args.output, image):
        raise OSError(f"Could not save image: {args.output}")
    print(f"Saved face-detected image to {args.output} ({len(faces)} face(s)).")


def exercise_6_18(args):
    cv2 = cv2_module()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the webcam.")
        if not cv2.imwrite(args.output, frame):
            raise OSError(f"Could not save webcam image: {args.output}")
        print(f"Saved webcam frame to {args.output}")
    finally:
        capture.release()


def exercise_6_19(args):
    cv2 = cv2_module()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (args.kernel, args.kernel),
                               sigmaX=args.sigma, sigmaY=args.sigma)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    if not cv2.imwrite(args.output, sketch):
        raise OSError(f"Could not save sketch: {args.output}")
    print(f"Saved pencil sketch to {args.output}. Larger kernels smooth more detail.")


def exercise_6_20(args):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install mediapipe before running this exercise.") from exc
    cv2 = cv2_module()
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=args.max_faces,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for index, filename in enumerate(args.images, 1):
            image = cv2.imread(filename)
            if image is None:
                raise FileNotFoundError(filename)
            result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            count = len(result.multi_face_landmarks or [])
            print(f"{filename}: {count} face(s)")
            if result.multi_face_landmarks:
                for landmarks in result.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=image, landmark_list=landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    )
            output = f"{Path(filename).stem}_mesh{Path(filename).suffix or '.png'}"
            if not cv2.imwrite(output, image):
                raise OSError(f"Could not save annotated image: {output}")
            print(f"Saved {output}")


def exercise_6_21(args):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install mediapipe before running this exercise.") from exc
    cv2 = cv2_module()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    try:
        with mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as mesh:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mesh.process(rgb)
                if results.multi_face_landmarks:
                    height, width = frame.shape[:2]
                    for face_number, face in enumerate(results.multi_face_landmarks, 1):
                        points = []
                        for landmark_index, point in enumerate(face.landmark):
                            x, y = int(point.x * width), int(point.y * height)
                            points.append((x, y))
                            print(f"face={face_number} landmark={landmark_index} x={x} y={y}")
                        for x, y in points:
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.imshow("Exercise 6.21 - press ESC to quit", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


EXERCISES = {
    "6.1": exercise_6_1, "6.2": exercise_6_2, "6.3": exercise_6_3,
    "6.4": exercise_6_4, "6.5": exercise_6_5, "6.6": exercise_6_6,
    "6.7": exercise_6_7, "6.8": exercise_6_8, "6.9": exercise_6_9,
    "6.10": exercise_6_10, "6.11": exercise_6_11, "6.12": exercise_6_12,
    "6.13": exercise_6_13, "6.14": exercise_6_14, "6.15": exercise_6_15,
    "6.16": exercise_6_16, "6.17": exercise_6_17, "6.18": exercise_6_18,
    "6.19": exercise_6_19, "6.20": exercise_6_20, "6.21": exercise_6_21,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=EXERCISES)
    parser.add_argument("--image")
    parser.add_argument("--compare-image")
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--images-dir", default=".")
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--video", help="Video filename (exercises 6.2 and 6.5)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--scale-factor", type=float, default=1.3)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--known-dir", default="known_faces")
    parser.add_argument("--tolerance", type=float, default=0.6)
    parser.add_argument("--recognizer", choices=("eigen", "fisher", "lbph"), default="lbph")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--model", default="trained_faces.yml")
    parser.add_argument("--database", default="database.txt")
    parser.add_argument("--id", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--course", default="")
    parser.add_argument("--address", default="")
    parser.add_argument("--source-image")
    parser.add_argument("--target-image")
    parser.add_argument("--output", default="output.jpg")
    parser.add_argument("--kernel", type=int, default=21)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--max-faces", type=int, default=5)
    args = parser.parse_args()
    if args.kernel < 1 or args.kernel % 2 == 0:
        parser.error("--kernel must be a positive odd number")
    if args.exercise in ("6.1", "6.3", "6.4", "6.7", "6.17", "6.19", "6.13", "6.14") and not args.image:
        parser.error("--image is required for this exercise")
    if args.exercise == "6.20" and not args.images:
        parser.error("--images requires one or more image filenames")
    if args.exercise in ("6.15", "6.16") and not (args.source_image and args.target_image):
        parser.error("--source-image and --target-image are required")
    if args.exercise == "6.13" and not args.compare_image:
        parser.error("--compare-image is required to test face verification")
    if args.exercise in ("6.11", "6.12") and not (args.id and args.name):
        parser.error("--id and --name are required")
    if args.exercise in ("6.2", "6.5") and not args.video:
        parser.error("--video is required; these exercises read from a video file")
    EXERCISES[args.exercise](args)


if __name__ == "__main__":
    main()
