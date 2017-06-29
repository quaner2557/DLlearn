dir = r"/media/quan/软件/AAAAA/SAR/BulkCarrier"

import cv2
import numpy as np
import os
import matplotlib

imagelist = os.listdir(dir)
ima = cv2.imread(r"/media/quan/软件/AAAAA/SAR/BulkCarrier/200805_001.bmp",0)
print(ima.shape)
arr = np.asarray(ima, dtype="float32")
print(arr)

win1 = cv2.namedWindow('show1',flags=0)
cv2.imshow('show1',ima)
cv2.waitKey(0)

imb = cv2.resize(ima, (256, 256), interpolation=cv2.INTER_CUBIC)
print(imb.shape)
win2 = cv2.namedWindow('show2',flags=0)
cv2.imshow('show2',imb)
cv2.waitKey(0)