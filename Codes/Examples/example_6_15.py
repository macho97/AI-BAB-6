import os
import sys
from deepface import DeepFace

# Fungsi bantuan agar DeepFace bisa menemukan gambar yang ada di dalam subfolder
def get_path(filename):
    if os.path.exists(filename):
        return filename
    path_in_codes = os.path.join("Codes", "Examples", filename)
    if os.path.exists(path_in_codes):
        return path_in_codes
    print(f"Error: Gambar '{filename}' tidak ditemukan!")
    sys.exit(1)

# Detect Faces =======================================================
print("1. Mendeteksi wajah pada oldwoman.jpg...")
# (Catatan: DeepFace v0.0.79+ mengganti detectFace menjadi extract_faces, kita gunakan try-except untuk kompatibilitas)
try:
    detected_face = DeepFace.detectFace(get_path("oldwoman.jpg"))
except AttributeError:
    detected_face = DeepFace.extract_faces(get_path("oldwoman.jpg"))
print("Wajah berhasil dideteksi!\n")

# Verify images ==========================================================
print("2. Memverifikasi kecocokan dua wajah...")
result = DeepFace.verify(get_path("face11a.jpg"), get_path("face11b.jpg"), model_name="Facenet")
print("Is verified: ", result["verified"], "\n")

# Get age, gender, race, emotion ====================================
print("3. Menganalisis demografi (umur, emosi, gender, ras)...")
demography = DeepFace.analyze(get_path("face10.jpg")) 

# Mendukung format return DeepFace baik list of dict maupun dict
if isinstance(demography, list):
    demography = demography[0]

print("Age: ", int(demography["age"]))
print("Emotion: ", demography["dominant_emotion"])

# Memastikan gender tercetak dengan benar (versi DeepFace baru me-return dictionary untuk gender)
gender = demography["gender"]
if isinstance(gender, dict):
    gender = max(gender, key=gender.get) # Ambil gender dengan persentase tertinggi
print("Gender: ", gender)

print("Race: ", demography["dominant_race"])