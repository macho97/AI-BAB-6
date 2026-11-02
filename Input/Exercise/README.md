# Chapter 6 exercises

Each exercise has a numbered Python entry point. Shared implementation is in
`chapter6_exercises.py`. For example:

```powershell
python .\Exercise\exercise_6_1.py --image .\Input\face.jpg
python .\Exercise\exercise_6_2.py --video .\Input\video.mp4
python .\Exercise\exercise_6_9.py --known-dir .\Input\known_faces
python .\Exercise\exercise_6_13.py --image .\Input\face.jpg --compare-image .\Input\other.jpg
python .\Exercise\exercise_6_14.py --image .\Input\face.jpg --model-dir .\Input\models
```

For the Streamlit database, button, camera, and slider tasks, run the dedicated apps:

```powershell
streamlit run .\Exercise\exercise_6_12.py
streamlit run .\Exercise\exercise_6_17.py
streamlit run .\Exercise\exercise_6_18.py
streamlit run .\Exercise\exercise_6_19.py
```

For CLI exercises, other inputs are supplied with `--help`. Examples that require `face_recognition`,
`opencv-contrib-python`, `deepface`, or `mediapipe` report the missing package
when selected. The webcam, image, video, and trained-model results depend on
the user's input files and hardware; no accuracy or speed measurements are
claimed without running those inputs.
