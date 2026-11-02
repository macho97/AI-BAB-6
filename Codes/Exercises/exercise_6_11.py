# Exercise 6.11
# Modify the previous Python program (Example 6.13) so that it also stores the address of the student.
import cv2
import numpy as np
import PySimpleGUI as sg
import sys
import os
from PIL import Image
import pandas as pd
import csv

dataPath = "data"
databaseFile = "database.txt"
register = False
recognizeFrame = False
sampleNum = 20
name = ""
Id = ""
course = ""
address = ""  # Menambahkan atribut address
faces = []
Id_list = []

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def createImages(frame, count):
    global dataPath, name, Id, detector
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(dataPath + "\\" + name + "." + Id + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
    return frame


def writeDatabase(databaseFile, row):
    # Menyimpan record siswa termasuk field Address
    if not os.path.exists(databaseFile):
        with open(databaseFile, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id", "Name", "Course", "Address"])
    with open(databaseFile, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    return "Saved to database"


def getImagesAndLabels(path):
    if not os.path.exists(path):
        return [], []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces_list = []
    ids_list = []
    for imagePath in imagePaths:
        extension = os.path.splitext(imagePath)[1]
        if extension != '.jpg':
            continue
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        try:
            student_id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces_list.append(imageNp)
            ids_list.append(student_id)
        except (IndexError, ValueError):
            continue
    return faces_list, ids_list


def Train(path):
    global recognizer
    faces_list, ids_list = getImagesAndLabels(path)
    if len(faces_list) == 0:
        return "No training data found"
    recognizer.train(faces_list, np.array(ids_list))
    recognizer.save("Trainner.yml")
    return "Training finished..."


def TrackImages(frame):
    global recognizer, detector
    if not os.path.exists("Trainner.yml") or not os.path.exists(databaseFile):
        return frame, "Model or database not found"
    recognizer.read("Trainner.yml")
    df = pd.read_csv(databaseFile, delimiter=',')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detector.detectMultiScale(gray, 1.2, 5)
    person_to_show = ""
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
        student_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf < 50:
            match_name = df.loc[df['Id'] == student_id]['Name'].values
            match_course = df.loc[df['Id'] == student_id]['Course'].values
            match_addr = df.loc[df['Id'] == student_id]['Address'].values if 'Address' in df.columns else ['']
            name_val = match_name[0] if len(match_name) > 0 else 'Unknown'
            course_val = match_course[0] if len(match_course) > 0 else ''
            addr_val = match_addr[0] if len(match_addr) > 0 else ''
            person = f"{student_id}-{name_val}-{course_val}-{addr_val}"
            person_to_show = f"{person} {int(conf)}%"
        else:
            person_to_show = f"Unknown {int(conf)}%"
    return frame, person_to_show


def main():
    global register, sampleNum, dataPath, name, Id, course, address, recognizeFrame
    sg.ChangeLookAndFeel('LightGreen')

    leftpanel = [
        [sg.Text('Face Recognition System (with Address)', size=(30, 1), justification='center', font='Helvetica 18', key='title')],
        [sg.Image(filename='', key='image')],
    ]
    rightpanel = [
        [sg.Text('Id:', size=(10, 1))],
        [sg.InputText("", key="Id")],
        [sg.Text('Name:', size=(10, 1))],
        [sg.InputText("", key="name")],
        [sg.Text('Course:', size=(10, 1))],
        [sg.InputText("", key="course")],
        [sg.Text('Address:', size=(10, 1))],
        [sg.InputText("", key="address")],  # Input field alamat siswa
        [sg.Button('1.Register', size=(15, 1), font='Helvetica 14')],
        [sg.Button('2.Train', size=(15, 1), font='Any 14')],
        [sg.Button('3.Recognize', size=(15, 1), font='Helvetica 14')],
    ]
    layout = [
        [
            sg.Column(leftpanel),
            sg.VSeperator(),
            sg.Column(rightpanel),
        ]
    ]

    window = sg.Window('Exercise 6.11 - Student Record with Address', location=(100, 100))
    window.Layout(layout).Finalize()

    info = ""
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        event, values = window.read(timeout=20, timeout_key='timeout')
        if event == sg.WIN_CLOSED:
            break
        elif event == '1.Register':
            register = True
            count = 0
        elif event == '2.Train':
            info = Train(dataPath)
        elif event == '3.Recognize':
            recognizeFrame = True

        name = values["name"]
        Id = values["Id"]
        course = values["course"]
        address = values["address"]

        ret, frame = cap.read()
        if not ret:
            continue
        if register:
            createImages(frame, count)
            info = "Saving " + str(count)
            count = count + 1
            if count > sampleNum:
                row = [Id, name, course, address]
                info = writeDatabase(databaseFile, row)
                register = False
        if recognizeFrame:
            frame, info = TrackImages(frame)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)
        window['title'].update(info)

    cap.release()
    cv2.destroyAllWindows()
    window.close()


if __name__ == '__main__':
    main()
