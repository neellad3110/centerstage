# importing the required libraries
import cv2
import math
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import shift
import time


# Define the video capture source (0 for webcam)
video_source = 0
image=[]
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


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def move_background_with_custom_ratio(original_frame, face_bbox, ratio_top_bottom=85, ratio_left_right=85):
    # Constants
    frame_height, frame_width, _ = original_frame.shape

    # Calculate background movement
    face_center_x = face_bbox[0] + face_bbox[2] / 2
    face_center_y = face_bbox[1] + face_bbox[3] / 2

    # Calculate the amount of background movement needed to keep the face in the center
    target_center_x = frame_width / 2
    target_center_y = frame_height / 2

    background_movement_x = target_center_x - face_center_x
    background_movement_y = target_center_y - face_center_y

    # Scale the background movement according to the specified ratios
    background_movement_x *= ratio_left_right / 100
    background_movement_y *= ratio_top_bottom / 100

    # Create a grid of coordinates for shifting
    x_coords = np.arange(frame_width)
    y_coords = np.arange(frame_height)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Perform background movement using scipy.ndimage.shift
    shifted_x = X - background_movement_x
    shifted_y = Y - background_movement_y

    background_shifted = np.zeros_like(original_frame)
    for channel in range(original_frame.shape[2]):
        background_shifted[:, :, channel] = shift(original_frame[:, :, channel], (background_movement_y, background_movement_x), mode='reflect')

    return background_shifted

# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore

# capturing the video feed from the camera
cam = cv2.VideoCapture(0)
latest_position=[]



while True:
    _, image = cam.read()

   
   

    #convert each frame from BGR to Grayscale
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces using Haar Cascade 
    face = haar_cascade.detectMultiScale(image, 1.3, 5)

    if(len(image)>0):

        # draw a rectangle around the face and update the text to Face Detected
        for (x, y, w, h) in face:

            

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.rectangle(image, (1050,110), (250, 630), (0, 255, 0), 2)

            center_x, center_y = x + w // 2, y + h // 2
            #sCalculate the distance from the screen edges to the face center
            # screen_center_x, screen_center_y = image.shape[1] // 2, image.shape[0] // 2
            
            # distance_left = calculate_distance((0, center_y), (center_x, center_y))  # Distance to left side
            # distance_right = calculate_distance((image.shape[1] - 1, center_y), (center_x, center_y))  # Distance to right side
            # distance_top = calculate_distance((center_x, 0), (center_x, center_y))  # Distance to top side
            # distance_bottom = calculate_distance((center_x, image.shape[0] - 1), (center_x, center_y))  # Distance to bottom side


            # Draw lines from the screen edges to the face center
            # cv2.line(image, (0, center_y), (center_x, center_y), (0, 255, 0), 2)          # Left side
            # cv2.line(image, (image.shape[1]-1, center_y), (center_x, center_y), (0, 255, 0), 2)  # Right side
            # cv2.line(image, (center_x, 0), (center_x, center_y), (0, 255, 0), 2)          # Top side
            # cv2.line(image, (center_x, image.shape[0]-1), (center_x, center_y), (0, 255, 0), 2)  # Bottom side
            
            face_bbox = [x, y, w, h]
            
            
            move_timer = time.localtime() # get struct_time
            time_string = int(time.strftime("%S", move_timer))
            
            if time_string % 5 == 0:

                image=move_background_with_custom_ratio(image,face_bbox)
                latest_position=face_bbox
            
            else:
                if(len(latest_position)>0):
                    image=move_background_with_custom_ratio(image,latest_position)
               
            
          
            #  # Extract the region of interest from the frame
            
            image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    else :
            image = image[roi_y1:roi_y2, roi_x1:roi_x2]
        

    image=cv2.flip(image,1)
    cv2.imshow("Center Stage", image)
    key = cv2.waitKey(10)

    # Press `q` to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()