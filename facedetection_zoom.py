import cv2
import math
import numpy as np
import time



video_source = 0
image=[]

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

    background_movement_x = (target_center_x - face_center_x) * (ratio_left_right / 100)
    background_movement_y = (target_center_y - face_center_y) * (ratio_top_bottom / 100)

    # Define the affine transformation matrix
    translation_matrix = np.float32([[1, 0, background_movement_x], [0, 1, background_movement_y]])

    # Apply the affine transformation using OpenCV
    background_shifted = cv2.warpAffine(original_frame, translation_matrix, (frame_width, frame_height), borderMode=cv2.BORDER_REFLECT)

    return background_shifted

# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore

# capturing the video feed from the camera
cam = cv2.VideoCapture(0)
latest_position=[]

last_shift_time = time.time()
background_shifted=None
transition_frames=10
while True:
    _, image = cam.read()

    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(image, 1.3, 5)

    if len(face) > 0:
        for (x, y, w, h) in face:
            face_bbox = [x, y, w, h]

            current_time = time.time()
            if current_time - last_shift_time >= 5:
                last_shift_time = current_time
                image = move_background_with_custom_ratio(image, face_bbox)
                latest_position = face_bbox

            elif background_shifted is not None:
                if transition_frames > 0:
                    alpha = 1.0 - (transition_frames / 15)  # Calculate interpolation factor
                    blended = cv2.addWeighted(image, alpha, background_shifted, 1 - alpha, 0)
                    transition_frames -= 1
                    image = blended
                else:
                    background_shifted = move_background_with_custom_ratio(background_shifted, latest_position)
                    image = background_shifted.copy()

                   
            else:
                if len(latest_position) > 0:
                    image = move_background_with_custom_ratio(image, latest_position)

            image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    else:
        image = image[roi_y1:roi_y2, roi_x1:roi_x2]

    image = cv2.flip(image, 1)
    
    cv2.imshow("Center Stage", image)
    key = cv2.waitKey(10)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()