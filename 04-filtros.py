#!/usr/bin/env python

# TODO: No pasar a gray o hacer todo en gray

import cv2   as cv
import numpy as np
import matplotlib.pyplot as plt
from umucv.stream import autoStream
import scipy.signal as signal
import scipy.ndimage.filters as fil
from umucv.util import ROI

filtro = 0
def nada(v):
    pass

cv.namedWindow("Input",cv.WINDOW_NORMAL)
cv.moveWindow('Input', 0, 0)
roi = ROI("Input")
color = True

for key,frame in autoStream():
    if ord('c') == key:
        color = not color
    if ord('x') == key:
        filtro = 0
        cv.destroyAllWindows()
        cv.namedWindow('Input',cv.WINDOW_AUTOSIZE)
        roi = ROI("Input")
        color = True

    if not color:
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,0,255), thickness=2)

        # A continuación, dependiendo del filtro a aplicar, creamos una ventana
        if ord('0') == key:
            cv.destroyAllWindows()
            filtro = 0
            cv.namedWindow('Input',cv.WINDOW_AUTOSIZE)
        
        if ord('1') == key:
            cv.destroyAllWindows()
            filtro = 1
            cv.namedWindow('Convolucion',cv.WINDOW_AUTOSIZE)
            cv.createTrackbar("a11","Convolucion",15,30,nada)
            cv.createTrackbar("a12","Convolucion",15,30,nada)
            cv.createTrackbar("a13","Convolucion",15,30,nada)
            cv.createTrackbar("a21","Convolucion",15,30,nada)
            cv.createTrackbar("a22","Convolucion",15,30,nada)
            cv.createTrackbar("a23","Convolucion",15,30,nada)
            cv.createTrackbar("a31","Convolucion",15,30,nada)
            cv.createTrackbar("a32","Convolucion",15,30,nada)
            cv.createTrackbar("a33","Convolucion",15,30,nada)
            cv.createTrackbar("r", "Convolucion",15,30,nada)

        if ord('2') == key:
            cv.destroyAllWindows()
            filtro = 2
            cv.namedWindow("Box",cv.WINDOW_NORMAL)
            cv.createTrackbar("horizontal", "Box", 5, 75, nada)
            cv.createTrackbar("vertical", "Box", 5, 75, nada)

        if ord('3') == key:
            cv.destroyAllWindows()
            filtro = 3
            cv.namedWindow("Gaussian",cv.WINDOW_NORMAL)
            cv.createTrackbar("sigma", "Gaussian", 5, 30, nada)

        if ord('4') == key:
            cv.destroyAllWindows()
            filtro = 4
            cv.namedWindow("Mediana",cv.WINDOW_NORMAL)
            cv.createTrackbar("tam", "Mediana", 7, 15, nada)

        if ord('5') == key:
            cv.destroyAllWindows()
            filtro = 5
            cv.namedWindow("Bilateral",cv.WINDOW_NORMAL)
            cv.createTrackbar("d","Bilateral",5,30,nada) # Diameter of pixel neighborhood
            cv.createTrackbar("sigmaColor","Bilateral",25,75,nada) # Distance in pixel value to group
            cv.createTrackbar("sigmaSpace","Bilateral",25,75,nada) # Distance in space for neighborhood grouping

        if ord('6') == key:
            cv.destroyAllWindows()
            filtro = 6
            cv.namedWindow("Min",cv.WINDOW_NORMAL)
            cv.createTrackbar("tam","Min",5,30,nada)

        if key == ord('7') == key:
            cv.destroyAllWindows()
            filtro = 7
            cv.namedWindow("Max",cv.WINDOW_NORMAL)
            cv.createTrackbar("tam","Max",5,30,nada)

        if 0 == filtro:
            cv.imshow('Input',frame)

        if 1 == filtro:
            a11 = cv.getTrackbarPos('a11','Convolucion')/5 - 3
            a12 = cv.getTrackbarPos('a12','Convolucion')/5 - 3
            a13 = cv.getTrackbarPos('a13','Convolucion')/5 - 3
            a21 = cv.getTrackbarPos('a21','Convolucion')/5 - 3
            a22 = cv.getTrackbarPos('a22','Convolucion')/5 - 3
            a23 = cv.getTrackbarPos('a23','Convolucion')/5 - 3
            a31 = cv.getTrackbarPos('a31','Convolucion')/5 - 3
            a32 = cv.getTrackbarPos('a32','Convolucion')/5 - 3
            a33 = cv.getTrackbarPos('a33','Convolucion')/5 - 3
            r = cv.getTrackbarPos('r','Convolucion')/15
            ker = np.array([[ a11, a12, a13]
                ,[ a21, a22, a23]
                ,[ a31, a32, a33]])
            ker = ker * r
            result = np.zeros_like(frame[y1:y2+1,x1:x2+1])
            if color:
                result[:,:,0] = signal.convolve2d(frame[y1:y2+1,x1:x2+1,0], ker, boundary='symm', mode='same')
                result[:,:,1] = signal.convolve2d(frame[y1:y2+1,x1:x2+1,1], ker, boundary='symm', mode='same')
                result[:,:,2] = signal.convolve2d(frame[y1:y2+1,x1:x2+1,2], ker, boundary='symm', mode='same')
            else:
                result = signal.convolve2d(frame[y1:y2+1,x1:x2+1], ker, boundary='symm', mode='same')
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Convolucion',frame)

        if 2 == filtro:
            n = cv.getTrackbarPos('horizontal','Box')+1
            m = cv.getTrackbarPos('vertical','Box')+1
            result = cv.boxFilter(frame[y1:y2+1,x1:x2+1],-1,(n,m))
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Box',frame)

        if 3 == filtro:
            sigma = cv.getTrackbarPos('sigma','Gaussian')+1
            result = cv.GaussianBlur(frame[y1:y2+1,x1:x2+1],(0,0),sigma)
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Gaussian',frame)

        if 4 == filtro:
            tam = 2*cv.getTrackbarPos('tam','Mediana')+1
            result = cv.medianBlur(frame[y1:y2+1,x1:x2+1],tam)
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Mediana',frame)

        if 5 == filtro:
            d = cv.getTrackbarPos('d','Bilateral')+1
            sigmaColor = cv.getTrackbarPos('sigmaColor','Bilateral')
            sigmaSpace = cv.getTrackbarPos('sigmaSpace','Bilateral')
            result = cv.bilateralFilter(frame[y1:y2+1,x1:x2+1],d,sigmaColor,sigmaSpace)
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Bilateral',frame)

        if 6 == filtro:
            tam = cv.getTrackbarPos('tam','Min')+1
            result = fil.minimum_filter(frame[y1:y2+1,x1:x2+1],tam)
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Min',frame)

        
        if 7 == filtro:
            tam = cv.getTrackbarPos('tam','Max')+1
            result = fil.maximum_filter(frame[y1:y2+1,x1:x2+1],tam)
            frame[y1:y2+1,x1:x2+1] = result
            cv.imshow('Max',frame)

    else:
        cv.imshow('Input',frame)
cv.destroyAllWindows()
