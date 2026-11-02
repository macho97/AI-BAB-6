# Exercise 6.14
# Modify the previous Python program (Example 6.15b), and try different photos for
# age, gender, race, and emotion detection. Comment on the results.
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

age_dict = {
    0: 'below_20',
    1: '20-30',
    2: '30-40',
    3: '40-50',
    4: 'above_50'
}

gender_dict = {
    0: 'Male',
    1: 'Female'
}

ethnicity_dict = {
    0: 'arab',
    1: 'asian',
    2: 'black',
    3: 'hispanic',
    4: 'indian',
    5: 'white'
}

emotion_dict = {
    0: 'angry',
    1: 'happy',
    2: 'neutral',
    3: 'sad'
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def gender_pred(image, model):
    gen = model.predict(image).argmax()
    return gender_dict[gen]


def emotions_pred(image, model):
    emotion = model.predict(image).argmax()
    return emotion_dict[emotion]


def ethnicity_pred(image, model):
    eth = model.predict(image).argmax()
    return ethnicity_dict[eth]


def age_pred(image, model):
    age = model.predict(image).argmax()
    return age_dict[age]


def predict_on_images(test_files, models):
    gender_model, ethnicity_model, age_model, emotions_model = models

    for file in test_files:
        if not os.path.exists(file):
            print(f"File {file} tidak ditemukan, lewati.")
            continue

        print(f"\nMemproses gambar: {file}")
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 10)
        print(f"Jumlah wajah ditemukan: {len(faces)}")

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (70, 70), interpolation=cv2.INTER_AREA)
            face = np.array(roi_color, dtype='uint8') / 255.0
            face = face.reshape(1, 70, 70, 3)

            g_pred = gender_pred(face, gender_model)
            e_pred = ethnicity_pred(face, ethnicity_model)
            a_pred = age_pred(face, age_model)
            emo_pred = emotions_pred(face, emotions_model)

            print(f"  Hasil Prediksi -> Usia: {a_pred}, Gender: {g_pred}, Etnis: {e_pred}, Emosi: {emo_pred}")

            cv2.putText(image, "Age:" + str(a_pred), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(image, "Gender: " + str(g_pred), (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(image, "Emotion: " + str(emo_pred), (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(image, "Ethnicity: " + str(e_pred), (x, y + h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        output_name = f"output_{os.path.splitext(file)[0]}.jpg"
        cv2.imwrite(output_name, image)
        cv2.imshow(f"Output - {file}", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        models = (
            load_model('gender_model.h5'),
            load_model('ethnicity_model.h5'),
            load_model('age_model.h5'),
            load_model('emotions_model.h5')
        )
        test_images = ['image2.jpeg', 'face10.jpeg', 'face0.jpg']
        predict_on_images(test_images, models)
    except Exception as e:
        print("Model file error:", e)
