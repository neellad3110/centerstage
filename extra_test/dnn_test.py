from imutils import face_utils
import dlib
import cv2


face_detect = dlib.get_frontal_face_detector()

video_capture = cv2.VideoCapture(0)
flag = 0

frames_to_skip = 20
frame_count = 0
while True:

    ret, frame = video_capture.read()


    frame_count += 20

    if frame_count % frames_to_skip != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detect(gray, 1)

    for (i, rect) in enumerate(rects):

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    frame=cv2.flip(frame,1)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()