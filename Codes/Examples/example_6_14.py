# Example 6.14 Face Recognition System face_recognition
# pip install face-recognition
import cv2
import numpy as np
import PySimpleGUI as sg
import sys
import os
import pandas as pd
import csv
import face_recognition

dataPath = "data"
databaseFile = "database.txt"
register = False
recognizeFrame = False
sampleNum = 1
name = ""
Id = ""
course = ""
faces = []
Id_list = []


def getFaces(path):
    if not os.path.exists(path):
        return [], []
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    count = 0
    for imagepath in imagepaths:
        ext = os.path.splitext(imagepath)[1].lower()
        if ext not in ('.jpg', '.jpeg'):
            continue

        face_image = face_recognition.load_image_file(imagepath)
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) < 1:
            continue
        faces.append(face_encoding[0])
        student_id = os.path.split(imagepath)[-1].split('.')[0]
        IDs.append(student_id)
        count = count + 1
    print(IDs)
    return IDs, faces


def createImages(frame, count):
    global dataPath, name, Id
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.imwrite(dataPath + "\\" + name + "." + Id + '.' + str(count) + ".jpg", frame)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    return frame


def writeDatabase(databaseFile, row):
    if not os.path.exists(databaseFile):
        with open(databaseFile, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id", "Name", "Course"])
    with open(databaseFile, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    return "Saved to database"


# For testing phase
def recogImages(frame, known_face_names, known_face_encodings):
    if not os.path.exists(databaseFile) or len(known_face_encodings) == 0:
        return frame, "No database or face encodings"
    rgb_frame = frame[:, :, ::-1]
    df = pd.read_csv(databaseFile, delimiter=',')
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    name = "Unknown"
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if len(matches) < 1:
            continue

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            cr = df.loc[df['Name'] == name]['Course'].values
            if len(cr) > 0:
                name = name + "-" + str(cr[0])
            print(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame, name


def main():
    global register, sampleNum, dataPath, name, Id, course, recognizeFrame
    sg.ChangeLookAndFeel('LightGreen')

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
        [sg.Button('2.Update', size=(15, 1), font='Any 14')],
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

    known_face_names, known_face_encodings = getFaces(dataPath)

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
        elif event == '2.Update':
            known_face_names, known_face_encodings = getFaces(dataPath)
            info = "Update done!"
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
            if count >= sampleNum:
                row = [Id, name, course]
                info = writeDatabase(databaseFile, row)
                register = False
        if recognizeFrame:
            frame, info = recogImages(frame, known_face_names, known_face_encodings)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)
        window['title'].update(info)

    cap.release()
    cv2.destroyAllWindows()
    window.close()


if __name__ == '__main__':
    main()
