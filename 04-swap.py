#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
import numpy as np

# Manejador para añadir los puntos de las regiones
def manejador(event, x, y, flags, param):
    global puntos1
    global puntos2
    global cuadrilatero
    if event == cv.EVENT_LBUTTONDOWN:
        if cuadrilatero == 1 and len(puntos1) < 4:
            puntos1 = puntos1 + [[x,y]]
            if len(puntos1) == 4: puntos1 = np.array(puntos1)
        if cuadrilatero == 2 and len(puntos2) < 4:
            puntos2 = puntos2 + [[x,y]]
            if len(puntos2) == 4: puntos2 = np.array(puntos2)

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)

real = np.array([
    [0,0],
    [100,0],
    [100,100],
    [0,100],
    ])
puntos1 = []
puntos2 = []
cuadrilatero = 0
for key,frame in autoStream():
    if key == ord('1'):
        print('Rellenando cuadrilatero 1')
        cuadrilatero = 1
    if key == ord('2'):
        print('Rellenando cuadrilatero 2')
        cuadrilatero = 2
    if key == ord('x'):
        puntos1 = []
        puntos2 = []
        cuadrilatero = 0
    for i in range(1,len(puntos1)):
        cv.line(frame,puntos1[i-1],puntos1[i],(255,0,0))
    if len(puntos1)==4: cv.line(frame,puntos1[3],puntos1[0],(255,0,0))
    for i in range(1,len(puntos2)):
        cv.line(frame,puntos2[i-1],puntos2[i],(0,255,0))
    if len(puntos2)==4: cv.line(frame,puntos2[3],puntos2[0],(0,255,0))
    # Pasamos a la parte interesante. Los cuadriláteros ya están definidos
    if len(puntos1) + len(puntos2) == 8:
        m,n,_ = frame.shape
        mask1 = np.zeros_like(frame)
        cv.fillPoly(mask1, pts=[puntos1], color=(255,255,255))
        mask1 = mask1 > 128
        mask2 = np.zeros_like(frame)
        cv.fillPoly(mask2, pts=[puntos2], color=(255,255,255))
        mask2 = mask2 > 128
        H,_ = cv.findHomography(puntos1, puntos2)
        Hinv = np.linalg.inv(H)
        Htrans = cv.warpPerspective(frame,H,(n,m))
        HinvTrans= cv.warpPerspective(frame,Hinv,(n,m))
        np.copyto(frame,HinvTrans, where=mask1)
        np.copyto(frame,Htrans, where=mask2)
        #frame[mask1 > 128] = cv.warpPerspective(frame,H,(m,n))
        # rec = cv.warpPerspective(frame[],H,(400,500))
        # cv.rectangle(rec,real[0],real[2],(255,0,0),2)
        cv.imshow('Htrans',Htrans)
        cv.imshow('HinvTrans',HinvTrans)
    cv.imshow('webcam',frame)
cv.destroyAllWindows()
