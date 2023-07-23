import cv2

def detect_faces(frame):
    """
    Detects faces in a frame using OpenCV's pre-trained face cascade classifier.

    Parameters:
        frame (numpy.ndarray): The input frame.

    Returns:
        list: A list of rectangles representing the bounding boxes of the detected faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    return faces

def closer_look(frame, zoom_factor, center_x, center_y):
    """
    Applies a closer look effect to a frame.

    Parameters:
        frame (numpy.ndarray): The input frame.
        zoom_factor (float): The zoom factor for the closer look effect.
        center_x (int): The x-coordinate of the center point for the closer look effect.
        center_y (int): The y-coordinate of the center point for the closer look effect.

    Returns:
        numpy.ndarray: The frame with the closer look effect.
    """
    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Calculate the new dimensions after zooming
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)

    # Calculate the region of interest (ROI) coordinates
    roi_x = max(0, center_x - new_width // 2)
    roi_y = max(0, center_y - new_height // 2)
    roi_x_end = min(width, roi_x + new_width)
    roi_y_end = min(height, roi_y + new_height)

    # Extract the region of interest from the frame
    roi = frame[roi_y:roi_y_end, roi_x:roi_x_end]

    # Resize the region of interest to the original frame size
    zoomed_roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)

    # Create a mask to exclude the zoomed-in region from the original frame
    mask = cv2.bitwise_not(cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2GRAY))

    # Apply the mask to the original frame
    frame_copy = cv2.bitwise_and(frame, frame, mask=mask)

    # Combine the zoomed-in region with the original frame
    frame_copy[roi_y:roi_y_end, roi_x:roi_x_end] = zoomed_roi

    return frame_copy

if __name__ == "__main__":
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    # Initialize parameters for the closer look effect
    zoom_factor = 2.0  # Zoom factor for closer look

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()

        # Detect faces in the frame
        faces = detect_faces(frame)

        if len(faces) > 0:
            # Get the coordinates of the first detected face
            (x, y, w, h) = faces[0]

            # Calculate the center coordinates of the face
            center_x = x + w // 2
            center_y = y + h // 2

            # Apply the closer look effect
            closer_look_frame = closer_look(frame, zoom_factor, center_x, center_y)

            # Display the frame with the closer look effect
            cv2.imshow("Closer Look", closer_look_frame)
        else:
            # If no faces are detected, display the original frame
            cv2.imshow("Closer Look", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()
