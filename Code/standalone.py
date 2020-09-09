import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


class GUI(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        w, h = 650, 400
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        self.pack()
        self.file = Button(self, text='Browse', command=self.choose)
        self.choose = Label(self, text="Choose file").pack()
        self.image = PhotoImage(file='sgsits.jpg')
        self.label = Label(image=self.image)
        self.resultText = StringVar()
        self.nameText = StringVar()
        self.resultLable = Label(textvariable=self.resultText, pady=15)
        self.nameLable = Label(textvariable=self.nameText, pady=15)
        self.nameLable.config(font=("Courier", 44))
        self.file.pack()
        self.label.pack()
        self.resultLable.pack()
        self.nameLable.pack()
        self.resultText.set("Result")

    def choose(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        ifile = filedialog.askopenfile(
            parent=self, mode='rb', title='Choose a file')
        path = ifile.name
        image = Image.open(path)
        image = image.resize((600, 200))

        self.image2 = ImageTk.PhotoImage(image)
        self.label.configure(image=self.image2)
        self.label.image = self.image2
        testimg = cv2.imread(path)
        gray = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
        id, confidence = recognizer.predict(gray)
        print("===========================================")
        print(100 - confidence)
        if ((100 - confidence) > 50):
            result = str(100 - confidence) + "% match"
            self.resultText.set(result)
            self.nameText
            names = ['None', 'Aman', 'Shyam']
            print(result)
            self.nameText.set(names[id])
        else:
            self.nameText.set("None")
            self.resultText.set("No maching found")


root = Tk()
app = GUI(master=root)
app.mainloop()
root.destroy()


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Aman', 'Shyam']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print(minW)
print(minH)
i = 0

img = cv2.imread("testimage/genuine-03.png")
images = []
count = 0
for filename in os.listdir("testimage"):
    testimg = cv2.imread(os.path.join("testimage", filename))
    # print(filename)
    count = count + 1
    gray = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
    id, confidence = recognizer.predict(gray)
    # print("===========================================")
    if((100 - confidence) > 80):
        print(str(count) + "\t" + filename + "\t\t\t" +
              str(id) + "\t" + str(100 - confidence))
        cv2.imshow(str(id) + ", " + filename, testimg)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if (confidence < 100):
    id = names[id]
    confidence = "  {0}%".format(round(100 - confidence))
else:
    id = "unknown"
    confidence = "  {0}%".format(round(100 - confidence))


exit()

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

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5),
                    font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
