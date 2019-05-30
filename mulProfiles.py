import numpy as np
import cv2, os
import sys

front_cascade = cv2.CascadeClassifier('cascades/cascade_front.xml')
side_cascade = cv2.CascadeClassifier('cascades/cascade_side.xml')
back_cascade = cv2.CascadeClassifier('cascades/cascade_back.xml')

path = sys.argv[1]
snaps = os.listdir(path)
i = 0

for i,snap in enumerate(snaps):
    img = cv2.imread(path + os.sep + snap)
    i += 1
    if(i % 10):
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_front = front_cascade.detectMultiScale(gray, 1.3, 5)
    face_side = side_cascade.detectMultiScale(gray, 1.3, 5)
    face_back = back_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face_front:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    for (x,y,w,h) in face_side:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    for (x,y,w,h) in face_back:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
