import cv2
from mtcnn.mtcnn import MTCNN

# Initialize the video capture
video_capture = cv2.VideoCapture(0)


# Initialize the MTCNN detector
detector = MTCNN()

while True:
    ret, frame = video_capture.read()

        # Detect faces in the image
    faces = detector.detect_faces(frame)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frame
    image=cv2.flip(frame,1)
    cv2.imshow('Video', image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
