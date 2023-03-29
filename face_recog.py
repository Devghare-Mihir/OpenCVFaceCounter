import cv2
import datetime
import time
# import numpy as np
# from deepface import DeepFace

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the font for text display
font = cv2.FONT_HERSHEY_SIMPLEX

start_time=time.time()
stopwatch_running= True
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        #emotions
        # try:
        #     analyze= DeepFace.analyze(frame, actions=['emotion'])
        #     # print(analyze)
        #     print(analyze['dominant_emotion'])
        # except:
        #     print("No face detected")
        
    # Display the number of faces detected
    num_faces = len(faces)
    cv2.putText(frame, "Face detected: " + str(num_faces), (10, 30), font, 1, (255, 20, 50), 2, cv2.LINE_AA)
    
    #CURRENT TIME
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %a %H:%M:%S")
    cv2.putText(frame, current_time, (400, 30), font, 0.5, (1, 1, 1), 1, cv2.LINE_AA)

    #TIME ELAPSED
    if stopwatch_running:
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
    
        # Display the elapsed time
        cv2.putText(frame, "Time Elapsed: "+str(int(elapsed_time)) + " sec", (450, 50), font, 0.5, (1, 1, 1), 1, cv2.LINE_AA)
    

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    
    # Break the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()