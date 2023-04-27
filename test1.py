import cv2
import argparse
import time
from threading import Thread
import mysql.connector
import requests
from mysql.connector.cursor import MySQLCursor
import datetime


#import mysql.connector

from compreface import CompreFace
from compreface.service import RecognitionService
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="teamdev"
)

# Create a cursor object
mycursor = mydb.cursor()
mycursor.execute("SELECT MAX(id) FROM utilisateur")
result = mycursor.fetchone()
new_id = result[0] + 1

sql = "INSERT INTO pointage (`date`,`heure_pointage`,`utilisateur_id`) VALUES (%s, %s, %s)"

# Get the current date and time in the desired format
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

# Set the values to be inserted
val = (timestamp, timestamp, "103")

# Execute the query with the values
mycursor = mydb.cursor()
mycursor.execute(sql, val)

# Commit the changes to the database
mydb.commit()

# Close the database connection
mydb.close()

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api-key", help="CompreFace recognition service API key", type=str, default='00000000-0000-0000-0000-000000000002')
    parser.add_argument("--host", help="CompreFace host", type=str, default='http://localhost')
    parser.add_argument("--port", help="CompreFace port", type=str, default='8000')

    args = parser.parse_args()

    return args

class ThreadedCamera:
    def __init__(self, api_key, host, port):
        self.active = True
        self.results = []
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        compre_face: CompreFace = CompreFace(host, port, {
            "limit": 0,
            "det_prob_threshold": 0.8,
            "prediction_count": 1,
            "face_plugins": "age,gender",
            "status": False
        })

        self.recognition: RecognitionService = compre_face.init_face_recognition(api_key)

        self.FPS = 1/30

        # Start frame retrieval thread
        self.thread = Thread(target=self.show_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    def show_frame(self):
        print("Started")
        while self.capture.isOpened():
            (status, frame_raw) = self.capture.read()
            self.frame = cv2.flip(frame_raw, 1)

            if self.results:
                results = self.results
                for result in results:
                    box = result.get('box')
                    age = result.get('age')
                    gender = result.get('gender')
                    mask = result.get('mask')
                    subjects = result.get('subjects')
                    if box:
                        cv2.rectangle(img=self.frame, pt1=(box['x_min'], box['y_min']),
                                      pt2=(box['x_max'], box['y_max']), color=(0, 255, 0), thickness=1)
                        if age:
                            age = f"Age: {age['low']} - {age['high']}"
                            cv2.putText(self.frame, age, (box['x_max'], box['y_min'] + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        if gender:
                            gender = f"Gender: {gender['value']}"
                            cv2.putText(self.frame, gender, (box['x_max'], box['y_min'] + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        if mask:
                            mask = f"Mask: {mask['value']}"
                            cv2.putText(self.frame, mask, (box['x_max'], box['y_min'] + 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                        if subjects:
                            subjects = sorted(subjects, key=lambda k: k['similarity'], reverse=True)
                            subject = f"Subject: {subjects[0]['subject']}"
                            similarity = f"Similarity: {subjects[0]['similarity']}"
                            cv2.putText(self.frame, subject, (box['x_max'], box['y_min'] + 75),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                            cv2.putText(self.frame, similarity, (box['x_max'], box['y_min'] + 95),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        else:
                            subject = f"No known faces"
                            cv2.putText(self.frame, subject, (box['x_max'], box['y_min'] + 75),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow('CompreFace demo', self.frame)
            time.sleep(self.FPS)

            if cv2.waitKey(1) & 0xFF == 27:
                self.capture.release()
                cv2.destroyAllWindows()
                self.active=False

    def is_active(self):
        return self.active

    def update(self):
        if not hasattr(self, 'frame'):
            return

        _, im_buf_arr = cv2.imencode(".jpg", self.frame)
        byte_im = im_buf_arr.tobytes()
        data = self.recognition.recognize(byte_im)
        self.results = data.get('result')


if __name__ == '__main__':
    args = parseArguments()
    threaded_camera = ThreadedCamera(args.api_key, args.host, args.port)
    while threaded_camera.is_active():
        threaded_camera.update()