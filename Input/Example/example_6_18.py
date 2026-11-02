"""Streamlit image-upload face detector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Exercise.exercise_6_17 import main

main()
