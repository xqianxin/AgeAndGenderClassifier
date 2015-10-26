import cv2
import sys
from CropFace import CropFace
from test import AgeAndGenderClassifier

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
video_capture = cv2.VideoCapture(0)

age_gender = AgeAndGenderClassifier()
age_gender.setModel()

isDetect = False
eye_left = ()
eye_right = ()
faces = ()
n = 0
while True:
    n = (n + 1) % 100
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0 and n == 0:
        n = (n - 1) % 100
    if n == 0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            isDetect = True
            if isDetect:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame[y:y+h,x:x+w]
                print age_gender.getAgeAndGender(img)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
video_capture.release()
cv2.destroyAllWindows()
