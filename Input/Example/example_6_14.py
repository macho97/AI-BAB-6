"""Webcam face recognition using the known image collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Exercise"))
from chapter6_exercises import main

sys.argv[1:1] = ["6.9"]
main()
