"""Run DeepFace's webcam-stream demo."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=".")
    args = parser.parse_args()
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise SystemExit("Install deepface before running this example.") from exc
    DeepFace.stream(db_path=args.database)


if __name__ == "__main__":
    main()
