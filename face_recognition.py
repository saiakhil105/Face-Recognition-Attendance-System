import cv2
import numpy as np
import os 
import mysql.connector

def updateinDB(id):
    print(id)
    q = str(id)
    mydb = mysql.connector.connect(host="localhost",user="root",passwd="Akhil@123",database="project")
    mycursor = mydb.cursor()
    mycursor.execute("update student set status = 'p' where id ="+q)
    mydb.commit()
    
    
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_TRIPLEX
id = 0
names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] 

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
   
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,255,0), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        updateinDB(id)
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()

