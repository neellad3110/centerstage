import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model for object detection
model_weights = '/Users/neellad/Documents/Projects and practice/CenterStage/extra_test/MobileNetSSD_deploy.caffemodel'
model_config = '/Users/neellad/Documents/Projects and practice/CenterStage/extra_test/MobileNetSSD_deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert the frame to a blob (input format required by the model)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the model
    net.setInput(blob)

    # Run forward pass through the model to get detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])

            # Check if the detected class is "person"
            if class_id == 15:
                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                (startX, startY, endX, endY) = box.astype(int)

                # Draw bounding box and label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f'Face: {confidence:.2f}'
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    # Display the frame
    image=cv2.flip(frame,1)
    cv2.imshow('Video', image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()