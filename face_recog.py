import cv2  
import flask
from deepface import DeepFace
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #importing face detecting opencv file
camera_port = 0
camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW) 

if not camera.isOpened():
  raise IOError("Webcam was unable to open! TRY again.") #raises IO error if web camera is not found

if not camera.isOpened():
  cap = cv2.VideoCapture(1) 

while True: #infinite loop untill pressed the desired button('q')
  res,frame = camera.read() #capture a particular frame

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False) #captured frame analysis through deepface library 
  
  faces = faceCascade.detectMultiScale(gray, 1.1, 3)

  for(x, y, w, h) in faces:     #creation of box around face 
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  font = cv2.FONT_HERSHEY_DUPLEX
  frame = cv2.flip(frame, 1)
  cv2.putText(frame,result['dominant_emotion'],(100, 100),font, 2,(0,0,255),2,cv2.LINE_AA)


  y = 100
  for x in result['emotion']:
    cv2.putText(frame,str(x) + ": " + str(int(result['emotion'][x])) + "%",(400, y),font, 1,(0, 234, 233),2,cv2.LINE_8)
    y = y + 30

  if(result['dominant_emotion'] == 'happy') :
    cv2.putText(frame,"Its amazing to see you happy!",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
  elif(result['dominant_emotion'] == 'sad') :
    cv2.putText(frame,"Don't be sad",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
  elif(result['dominant_emotion'] == 'disgust') :
    cv2.putText(frame,"This is very bad",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
  elif(result['dominant_emotion'] == 'fear') :
    cv2.putText(frame,"I think there is a ghost!",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
  elif(result['dominant_emotion'] == 'angry') :
    cv2.putText(frame,"Please calm down!",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
  elif(result['dominant_emotion'] == 'neutral') :
    cv2.putText(frame,"Try other emotions too!",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
  elif (result['dominant_emotion'] == 'surprised') :
    cv2.putText(frame,"Woah! what was that?",(50, 450),font, 1,(255,128,0),2,cv2.LINE_4)
    
  cv2.imshow('Emotion Detector', frame)

  if cv2.waitKey(2) & 0xFF == ord('q'):
    break

camera.release()
cv2.destroyAllWindows()
