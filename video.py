import numpy as np
import cv2 as cv
from Step1 import api as register
from Step2 import api as detector
from Step2.func import Box
from typing import List, Tuple

cap = cv.VideoCapture('data/test.mp4')

cnt = cap.get(cv.CAP_PROP_FRAME_COUNT)

print(cnt)

cnt = cap.set(cv.CAP_PROP_POS_FRAMES, 10.0)
ret, frame = cap.read()

img1 = frame
img1 = frame[60:900, 44:500:, :]

cnt = cap.set(cv.CAP_PROP_POS_FRAMES, 150.0)

cv.imshow('base', img1)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # pos = cap.get(cv.CAP_PROP_POS_FRAMES)
    # print(pos)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    img2 = frame[60:900, 44:500:, :]
    base, target = register(img1, img2)

    cv.imshow('base', base)
    img3 = img2
    p = 1.0
    img3[:,:,2] = base[:,:,2] * p + (1-p)* img3[:,:,2]

    diff, boxes = detector(base, target, expand_p=1.2)

    # diff = diff.astype(np.uint8)
    # diff = cv.cvtColor(diff, cv.COLOR_GRAY2RGB)

    # print(diff.shape)
    for box in boxes:
        #img2 = cv.rectangle(img2, (box.xmin, box.ymin), (box.xmax, box.ymax), (255,255,0), thickness=2)
        diff = cv.rectangle(diff, (box.xmin, box.ymin), (box.xmax, box.ymax), (255,0,0), thickness=0)
    cv.imshow('diff', diff)
    cv.imshow('frame', img3)
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()