#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import opencv library
import cv2

#include desired haar-cascades
#face, eye and smile haar-cascades are used, which after downloading need to be placed in the working directory.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')

#user enters program CLI 
#user can choose between 2 options, Real-time or Upload Photo
print("WELCOME TO SMILE DETECTOR SYSTEM")
print("\n*****OPTIONS*****\n")
print("1) Real-time Webcam Smile Detector")
print("2) Uploaded Photo Smile Detector")

option = int(input("Choice(1 or 2): "))

#Real-time smile detector
if option==1:
    print("Press 'q' to quit the program")
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    def detect(gray, frame): 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #to detect the face
        for (x, y, w, h) in faces: #face data is stored as tuples of coordinates
            
            #a rectangular box appears when face is detected
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
            
            #The roi_gray defines the region of interest of the face and 
            #roi_color does the same for the original frame. 
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w] 
           
            #smile detection is applied using the cascade
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

            # "Smiling" text is shown when smiling face is detected
            if len(smiles) > 0:
                cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 255, 255))

            # a rectangular box appears when smile is detected
            for (sx, sy, sw, sh) in smiles: 
                cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
        return frame

    #User's webcam is used to capture real-time video
    video_capture = cv2.VideoCapture(1) 

    while True: 
            # Captures video_capture frame by frame 
            _, frame = video_capture.read()
            # To capture image in monochrome 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calls the detect() function 
            canvas = detect(gray, frame)
            # Displays the result on camera feed 
            cv2.imshow('Real-time Smile Detector', canvas)
            # The control breaks once q key is pressed 
            if cv2.waitKey(1) & 0xff == ord('q'): 
                break
            

    # Release the capture once all the processing is done.
    video_capture.release()                                
    cv2.destroyAllWindows()

else:
    
    #User is prompted to enter their image path directory
    path = input("Enter your image path: ")

    #This is My path : '/Users/aimanwafiqbinappandi/Desktop/WIX3001/Individual Assignement/Aiman Wafiq bin Appandi.jpg'
    
    image = cv2.imread(path) #To read image 
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml') #Smile Haar-cascades is used 
    smiles  = smile_cascade.detectMultiScale(image, scaleFactor = 1.8, minNeighbors = 20) #To detect smile

    #Rectangular box appears when smile is detected
    for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(image, (sx, sy), ((sx + sw), (sy + sh)), (0, 255,0), 5)

    #Output for smile detection is displayed
    cv2.imshow("Uploaded Image Smile Detector", image)

    # Release the capture once all the processing is done.
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

