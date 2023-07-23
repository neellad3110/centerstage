# import cv2
# import numpy as np

# # Define the coordinates of the top-left corner and the bottom-right corner of the region of interest
# top_left = (200, 110)
# bottom_right = (1050, 630)

# # Initialize the video capture object
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture the frame from the video
#     ret, frame = cap.read()
    
#     # Crop the frame to the region of interest
#     roi = frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
#     # Check the size of the ROI
#     width, height = roi.shape[:2]
#     roi=cv2.flip(frame,1)
#     print(height," - ",width)

#     # Display the frame and the region of interest
#     # cv2.imshow("Video", frame)
#     cv2.imshow("ROI", roi)

#     # Press `q` to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object
# cap.release()

# # Close all the windows
# cv2.destroyAllWindows()

import cv2
import numpy as np

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

# Get the width and height of the video capture
width = int(cap.get(3))
height = int(cap.get(4))

# Define the region of interest (ROI) coordinates
# You can adjust these coordinates according to your desired region
roi_x1, roi_y1 = int(width * 0.15), int(height * 0.15)
roi_x2, roi_y2 = int(width * 0.85), int(height * 0.85)

print("(",roi_x1,",",roi_y1,") and (",roi_x2,",",roi_y2,")")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame=cv2.flip(frame,1)
    if not ret:
        print("Video capture ended.")
        break

    # Extract the region of interest from the frame
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Display the ROI frame
    cv2.imshow('Region of Interest', roi)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer, and close the windows
cap.release()
cv2.destroyAllWindows()
