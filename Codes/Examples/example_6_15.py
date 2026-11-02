# Example 6.15 The deepfacedemo.py Program
from deepface import DeepFace

# Detect Faces =======================================================
detected_face = DeepFace.detectFace("oldwoman.jpeg")

# Verify images ==========================================================
result = DeepFace.verify("face11a.jpeg", "face11b.jpeg")
print("Is verified: ", result["verified"])

# Get age, gender, race, emotion ====================================
# demography = DeepFace.analyze("perry1.jpg") # passing nothing as 2nd argument will find everything
# demography = DeepFace.analyze("face5.jpeg", ['age', 'gender', 'race', 'emotion'])
demography = DeepFace.analyze("face10.jpeg")  # passing nothing as 2nd argument will find everything

# Mendukung format return DeepFace baik list of dict maupun dict
if isinstance(demography, list):
    demography = demography[0]

print("Age: ", int(demography["age"]))
print("Emotion: ", demography["dominant_emotion"])
print("Gender: ", demography["gender"])
print("Race: ", demography["dominant_race"])
