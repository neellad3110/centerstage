# importing the required libraries
import cv2


# Define the video capture source (0 for webcam)
video_source = 0

# Initialize video capture
cap = cv2.VideoCapture(video_source)

# Check if the camera/video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

width = int(cap.get(3))
height = int(cap.get(4))

# Define the region of interest (ROI) coordinates
#  coordinates according to get ROI 
roi_x1, roi_y1 = int(width * 0.15), int(height * 0.15)
roi_x2, roi_y2 = int(width * 0.85), int(height * 0.85)


# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore

# capturing the video feed from the camera
cam = cv2.VideoCapture(0)
while True:
    _, image = cam.read()

    

     # Extract the region of interest from the frame
    image = image[roi_y1:roi_y2, roi_x1:roi_x2]


    #convert each frame from BGR to Grayscale
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces using Haar Cascade 
    face = haar_cascade.detectMultiScale(image, 1.3, 5)

    # draw a rectangle around the face and update the text to Face Detected
    for (x, y, w, h) in face:

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.rectangle(image, (1050,110), (250, 630), (0, 255, 0), 2)

    image=cv2.flip(image,1)
    cv2.imshow("Center Stage", image)
    key = cv2.waitKey(10)

    # Press `q` to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()