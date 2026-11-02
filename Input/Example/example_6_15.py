"""DeepFace face detection, image verification, and demographic analysis."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--compare-with")
    args = parser.parse_args()
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise SystemExit("Install deepface before running this example.") from exc
    detections = DeepFace.extract_faces(img_path=args.image, enforce_detection=True)
    print(f"Detected {len(detections)} face(s).")
    if args.compare_with:
        result = DeepFace.verify(args.image, args.compare_with)
        print(f"Verified match: {result['verified']}")
    print(DeepFace.analyze(
        img_path=args.image, actions=["age", "gender", "race", "emotion"]
    ))


if __name__ == "__main__":
    main()
