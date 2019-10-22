import numpy as np
import cv2, os
import sys

front_cascade = cv2.CascadeClassifier('cascades/cascade_front.xml')
side_cascade = cv2.CascadeClassifier('cascades/cascade_side.xml')
back_cascade = cv2.CascadeClassifier('cascades/cascade_back.xml')

path = sys.argv[1]
snaps = os.listdir(path)
snaps = sorted(snaps)
snaps.pop(0)
i = 0

for i,snap in enumerate(snaps):
    img = cv2.imread(path + os.sep + snap)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_front = front_cascade.detectMultiScale(gray, 1.3, 5)
    face_side = side_cascade.detectMultiScale(gray, 1.3, 5)
    face_back = back_cascade.detectMultiScale(gray, 1.3, 5)

    if len(face_front) > 0:
        for (x,y,w,h) in face_front:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv2.putText(img, 'front', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    if len(face_side) > 0:
        for (x,y,w,h) in face_side:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv2.putText(img, 'side', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    if len(face_back) > 0:
        for (x,y,w,h) in face_back:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv2.putText(img, 'back', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite('out/outim{:05d}.png'.format(i), img)
    i += 1
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
