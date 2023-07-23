# importing the required libraries
import cv2
import mediapipe as mp

mpfacedetect=mp.solutions.face_detection
mpdraw=mp.solutions.drawing_utils

detectface=mpfacedetect.FaceDetection()
# loading the haar case algorithm file into alg variable
alg = 'haarcascade_frontalface_default.xml'


# passing the algorithm to OpenCV
#haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



# capturing the video feed from the camera
cam = cv2.VideoCapture(0)
#cam.set(cv2.CAP_PROP_FPS, 60)
while True:
    _, image = cam.read()

    text = "Face not detected"

    # convert each frame from BGR to Grayscale
    #grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result=detectface.process(rgbImg)
    print(result);

    image=cv2.flip(image,1)
    cv2.imshow("Face Detection", image)


