import cv2
import numpy as np
import os
import sys

print(sys.version)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
eye_cascade = cv2.CascadeClassifier('lefteye_2splits.xml')
print("zero")
smile_cascade = cv2.CascadeClassifier('smile.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
print("one")
# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Aman','Shyam','Samidha']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print(minW)
print(minH)
i = 0;

i = i + 1
while True:
    ret, img = cam.read()
    #     img = cv2.flip(img, -1) # Flip vertically
    
    i = i + 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=10,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:

        start_point = (x, y)
        end_point = (x + w, y + h)
        top_point = (x+h, y)
        top_right_point = (x, y+w)
        top_right_end_point = (x +h , y)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        yellow_color = (0, 255, 255) 
        #cv2.line(img, start_point, end_point, color, 2) 

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        

        
        for (ex,ey,ew,eh) in eyes:
            startx = ex+ int((ew/2)) + x
            starty = ey+ int((eh/2)) + y
            eyecordiante = (startx, starty)
            cv2.line(img, eyecordiante,end_point , yellow_color, 2)
            cv2.line(img, eyecordiante,top_point , (255,255,0), 2)
            cv2.line(img, eyecordiante,top_right_point, yellow_color, 2)
            cv2.line(img, eyecordiante,start_point , (255,255,0), 2)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        print("==============================")
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in smile:
            startx = sx+ int((sw/2)) + x
            starty = sy+ int((sh/2)) + y
            
            eyecordiantesmile = (startx, starty)
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
            
            cv2.line(img, eyecordiantesmile, end_point , (0,0,255), 2)
            
            cv2.line(img, eyecordiantesmile, top_right_point, (0,0,255), 2)
            break
    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
