# Exercise 6.13
# Modify the previous Python program (Example 6.15), try different photos for face
# detection, face identification, and age, gender, race, and emotion detection.
# Comment on the results.
import os
from deepface import DeepFace

# Daftar foto yang diuji untuk deteksi demografi
test_photos = ["face10.jpeg", "face0.jpg", "face1.jpeg", "oldwoman.jpeg"]

print("=== 1. Deteksi Wajah & Analisis Demografi pada Berbagai Foto ===")
for photo in test_photos:
    if not os.path.exists(photo):
        print(f"File {photo} tidak ditemukan, lewati.")
        continue

    print(f"\n--- Analisis untuk: {photo} ---")
    try:
        # Analisis usia, gender, ras, dan emosi
        demography = DeepFace.analyze(photo, actions=['age', 'gender', 'race', 'emotion'])
        if isinstance(demography, list):
            demography = demography[0]

        print("  Age      : ", int(demography["age"]))
        print("  Gender   : ", demography["gender"])
        print("  Emotion  : ", demography["dominant_emotion"])
        print("  Race     : ", demography["dominant_race"])
    except Exception as e:
        print(f"  Gagal menganalisis {photo}: {e}")

# Verifikasi dua gambar untuk pencocokan wajah
print("\n=== 2. Verifikasi Pasangan Wajah ===")
pair = ("face11a.jpeg", "face11b.jpeg")
if os.path.exists(pair[0]) and os.path.exists(pair[1]):
    try:
        result = DeepFace.verify(pair[0], pair[1])
        print(f"Memverifikasi {pair[0]} dan {pair[1]}:")
        print("  Is verified: ", result["verified"])
        print("  Distance   : ", result.get("distance"))
    except Exception as e:
        print("Gagal verifikasi:", e)
else:
    print(f"File pasangan {pair} tidak lengkap.")
