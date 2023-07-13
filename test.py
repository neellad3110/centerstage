import cv2
import numpy as np

# Open the video capture object for webcam
cap = cv2.VideoCapture(0)

# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Unable to open the webcam.")
    exit()

# Read the first frame to get the video dimensions
ret, frame = cap.read()
if not ret:
    print("Unable to read the frame.")
    exit()

# Define the region of interest (ROI) for focus
x, y, width, height = 500,200 ,500, 500
i=0
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Unable to read the frame.")
        break
    



    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    r, frame = cap.read()
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(10)
    

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
