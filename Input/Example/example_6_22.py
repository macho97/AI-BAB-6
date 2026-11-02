"""Draw MediaPipe face-mesh contours on one or more images."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install mediapipe before running this example.") from exc
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, min_detection_confidence=0.5
    ) as mesh:
        for path in args.images:
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(path)
            result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for face in result.multi_face_landmarks or []:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, face, mp.solutions.face_mesh.FACEMESH_CONTOURS
                )
            output = f"{Path(path).stem}_mesh.png"
            if not cv2.imwrite(output, image):
                raise OSError(f"Could not save {output}")
            print(f"{path}: {len(result.multi_face_landmarks or [])} face(s), saved {output}")


if __name__ == "__main__":
    main()
