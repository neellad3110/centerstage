import cv2

# Load the pre-trained model
model_path = "deploy.prototxt"
#weights_path = "model.caffemodel"
weights_path = "model.xml"
try:
    net = cv2.dnn.readNet(model_path, weights_path)
except cv2.error as e:
    print(f"Error loading model files: {e}")
    exit()

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Resize frame for faster processing
    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

    # Convert frame to blob for input to the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    # Process the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * frame.shape[1]
            (startX, startY, endX, endY) = box.astype(int)

            # Draw the bounding box and confidence
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
