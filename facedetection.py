# importing the required libraries
import cv2

# loading the haar case algorithm file into alg variable
alg = 'haarcascade_frontalface_default.xml'


# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# capturing the video feed from the camera
cam = cv2.VideoCapture(0)
while True:
    _, image = cam.read()

    text = "Face not detected"

    # convert each frame from BGR to Grayscale
    #grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces using Haar Cascade 
    face = haar_cascade.detectMultiScale(image, 1.3, 5)

    # draw a rectangle around the face and update the text to Face Detected
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    
    image=cv2.flip(image,1)
    cv2.imshow("Face Detection", image)
    key = cv2.waitKey(10)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()