# Example 6.13 Face Recognition System OpenCV
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
faces = []
Id_list = []

# Local Binary Pattern Histogram is a Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.EigenFaceRecognizer_create()
# recognizer = cv2.face.FisherFaceRecognizer_create()

# creating detector for faces
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def createImages(frame, count):
    global dataPath, name, Id, detector
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(dataPath + "\\" + name + "." + Id + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
    return frame


def writeDatabase(databaseFile, row):
    # Create a student record file if does not exist
    if not os.path.exists(databaseFile):
        with open(databaseFile, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id", "Name", "Course"])
    with open(databaseFile, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    return "Saved to database"


def getImagesAndLabels(path):
    if not os.path.exists(path):
        return [], []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        extension = os.path.splitext(imagePath)[1]
        if extension != '.jpg':
            continue
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        try:
            student_id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(student_id)
        except (IndexError, ValueError):
            continue
    return faces, Ids


def Train(path):
    global recognizer, detector
    faces, Id = getImagesAndLabels(path)
    if len(faces) == 0:
        return "No training data found"
    recognizer.train(faces, np.array(Id))
    recognizer.save("Trainner.yml")
    return "Training finished..."


# For testing phase
def TrackImages(frame):
    global recognizer, detector
    if not os.path.exists("Trainner.yml") or not os.path.exists(databaseFile):
        return frame, "Model or database not found"
    recognizer.read("Trainner.yml")
    df = pd.read_csv(databaseFile, delimiter=',')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)
    person_to_show = ""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
        student_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        # If confidence is less than 100 ==>"0" perfect match
        if conf < 50:
            match_name = df.loc[df['Id'] == student_id]['Name'].values
            match_course = df.loc[df['Id'] == student_id]['Course'].values
            person = f"{student_id}-{match_name[0] if len(match_name) > 0 else 'Unknown'}-{match_course[0] if len(match_course) > 0 else ''}"
            person_to_show = f"{person} {int(conf)}%"
        else:
            person_to_show = f"Unknown {int(conf)}%"
    return frame, person_to_show


def main():
    global register, sampleNum, dataPath, name, Id, course, recognizeFrame
    sg.ChangeLookAndFeel('LightGreen')

    # define the window layout
    leftpanel = [
        [sg.Text('Face Recognition', size=(20, 1), justification='center', font='Helvetica 20', key='title')],
        [sg.Image(filename='', key='image')],
    ]
    rightpanel = [
        [sg.Text('Id:', size=(10, 1))],
        [sg.InputText("", key="Id")],
        [sg.Text('Name:', size=(10, 1))],
        [sg.InputText("", key="name")],
        [sg.Text('Course:', size=(10, 1))],
        [sg.InputText("", key="course")],
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

    window = sg.Window('Face Recognition System', location=(100, 100))
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

        ret, frame = cap.read()
        if not ret:
            continue
        if register:
            createImages(frame, count)
            info = "Saving " + str(count)
            count = count + 1
            if count > sampleNum:
                row = [Id, name, course]
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
