"""Draw MediaPipe face-mesh contours over webcam frames."""

import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("Install mediapipe before running this example.") from exc
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    try:
        with mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as mesh:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Could not read a webcam frame.")
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)
                for face in result.multi_face_landmarks or []:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, face, mp.solutions.face_mesh.FACEMESH_CONTOURS
                    )
                cv2.imshow("MediaPipe FaceMesh - ESC to exit", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
