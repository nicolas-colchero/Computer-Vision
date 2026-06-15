#!/usr/bin/env python
import numpy             as np
import cv2               as cv
import matplotlib.pyplot as plt
from umucv.stream import autoStream

def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY)
def grey(x: bool):
    if x.any(): return 255
    else: return 0

last_frame = None
mask = None
epsilon = 10
kernel = np.ones((5,5),np.uint8)

for key,frame in autoStream():
    if key==ord('x'):
        last_frame = None
        mask = None
    if last_frame is None:
        last_frame = bgr2gray(frame)
        continue
    delta = bgr2gray(frame) - last_frame
    if last_frame is not None and mask is None:
        mask = np.array([np.abs(x) < epsilon for x in delta],dtype=np.uint8)
        mask = cv.dilate(mask,kernel,iterations=1)
        last_frame = bgr2gray(frame)
        continue
    new_mask = np.array([np.abs(x) < epsilon for x in delta],dtype=np.uint8)
    new_mask = cv.dilate(new_mask,kernel,iterations=1)
    m,n = new_mask.shape
    cv.imshow('frame',frame)

    mask = np.logical_and(mask,new_mask,mask)
    dst = cv.inpaint(frame,mask.astype(np.uint8),1,cv.INPAINT_NS)
    cv.imshow('dst',dst)
    last_frame = bgr2gray(frame)
#    cv.imshow('last_frame',last_frame)
    
cv.destroyAllWindows()