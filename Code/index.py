
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
import time

# import training

class Application:
    def __init__(self, output_path = "./"):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.startSmaplling = 0;
        self.runsamping = 0;
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera
        self.face_detector = cv2.CascadeClassifier('frontalface_default.xml')
        self.root = tk.Tk()  # initialize root window
        self.root.title("Image Prosessing")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(padx=10, pady=10)



        # create a button, that when pressed, will take the current frame and save it to file
        lable1 = tk.Label(self.root, text="Enter ID")
        lable1.pack()
        self.id = tk.Entry(self.root)
        self.id.pack()
        btn = tk.Button(self.root, text="Get Sample!", command= self.startSmapling)
        btn.pack(fill="both", expand=True, padx=10, pady=10)
        btn1 = tk.Button(self.root, text="Start Face Recognition", command= self.startReco)
        btn1.pack(fill="both", expand=True, padx=10, pady=10)
        
        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok: # frame captured without any errors




            if(self.startSmaplling == 0):
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
                self.current_image = Image.fromarray(cv2image)  # convert image for PIL
                imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
                self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
                self.panel.config(image=imgtk)  # show the image
            elif(self.runsamping == 0):
                
               
                personIDtext  = self.id.get()

                try: 
                    int(personIDtext )
                    
                except ValueError:
                    print("invalid value ")
                    self.root.after(30, self.video_loop)
                    self.startSmaplling = 0
                    return
                self.personID = int(personIDtext)
                print(self.personID)
                self.count = 0;
                self.face_id = self.personID
                self.runsamping = 1
            if(self.runsamping == 1):
                #if(self.personID <= 0 and self.personID > 10):
                #    self.face_id = self.personID
                #    self.id.delete(0, 100)
                #    print("invalid value ")
                #    self.root.after(30, self.video_loop)
                #    self.startSmaplling = 0
                #    return

                ok, frame = self.vs.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                self.current_image = Image.fromarray(gray)  # convert image for PIL
                imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
                self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
                self.panel.config(image=imgtk)  # show the image
                #cv2.imshow('image', frame)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)     
                    self.count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/User." + str(self.face_id) + '.' + str(self.count) + ".jpg", gray[y:y+h,x:x+w])
                    #cv2.imshow('image', frame)
                    self.current_image = Image.fromarray(frame)  # convert image for PIL
                    imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
                    self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
                    self.panel.config(image=imgtk)  # show the image
                    print(self.count)
                    time.sleep(0.250)
                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    self.runsamping = 0                
                    self.startSmaplling = 0
                    
                elif self.count >= 30: # Take 30 face sample and stop video
                    self.runsamping = 0                
                    self.startSmaplling = 0
                    # execfile('training.py')
                    exec(open('training.py').read())
                    

        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def startSmapling(self):
        print("Starting Sampling")
        self.startSmaplling = 1
    def startReco(self):

        print("reco Start")
        # execfile('recognition.py')
        exec(open('recognition.py').read())

    def take_snapshot(self):

        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
        p = os.path.join(self.output_path, filename)  # construct output path
        self.current_image.save(p, "JPEG")  # save image as jpeg file
        print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
    help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()