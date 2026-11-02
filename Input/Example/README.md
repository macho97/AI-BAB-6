# Chapter 6 examples

Numbered programs are organized by the example numbers in the pasted material.
Examples 6.1-6.9 are available through `chapter6_examples.py`; later programs
are separate files because they use different libraries and model inputs.
OpenCV cascade XML files are loaded from OpenCV's installed data directory.

Examples:

```powershell
python .\Example\chapter6_examples.py --example 6.1 --image .\Input\face.jpg
python .\Example\chapter6_examples.py --example 6.2 --source .\Input\video.mp4
python .\Example\example_6_6.py --images .\Input\faces1.jpg .\Input\faces2.jpg
python .\Example\chapter6_examples.py --example 6.7 --image .\Input\face.jpg
python .\Example\example_6_10.py --dataset .\Input\faces --recognizer lbph
python .\Example\example_6_11.py --image .\Input\test.jpg
python .\Example\example_6_13.py register --id 1 --name "Student" --course AI --address "Surabaya"
python .\Example\example_6_13.py train
python .\Example\example_6_13.py recognize
python .\Example\example_6_15.py --image .\Input\face.jpg
python .\Example\example_6_22.py .\Input\face1.jpg .\Input\face2.jpg
```

Examples have per-file help where applicable. Install only the dependencies
needed for the program you run: `opencv-python`, `opencv-contrib-python`,
`face-recognition`, `deepface`, `tensorflow`, `mediapipe`, `Pillow`, and
`streamlit`. Some examples require pretrained model files and a webcam; those
assets were not included in the workspace.
