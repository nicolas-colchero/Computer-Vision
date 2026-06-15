#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
import numpy as np
import numpy.linalg as la

## Recibimos 3 puntos que suponemos alineados y equidistantes
def manejador(event, x, y, flags, param):
    global puntos
    global frame
    if event == cv.EVENT_LBUTTONDOWN:
        if len(puntos) < 3:
            puntos = puntos + [np.array([x,y])]
        if len(puntos)==2:
            v=puntos[0]-puntos[1]
            h,w = frame.shape[:2]
            if v[0]!=0:
                m=v[1]/v[0]
                x0=np.array([0,-(puntos[0][0]-0)*m+puntos[0][1]]).astype(np.int32)
                x1=np.array([w,-(puntos[1][0]-w)*m+puntos[1][1]]).astype(np.int32)
            else:
                x0=np.array([puntos[0][0],0]).astype(np.uint32)
                x1=np.array([puntos[0][0],h]).astype(np.uint32)
            print(x0,x1,m)
            cv.line(frame,x0,x1,color=(255,0,0))
            cv.imshow('webcam',frame)
frame = cv.imread('./images/CR/poles.jpg')
puntos = []
n = 20
cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)

def siguiente_punto(a,b):
    return (a*b+b*b)/(3*a-b)
## En la memoria se explica esta función
def punto_fuga(a,b):
    assert(b<a and a>0 and b>0)
    return b*(b+a)/(a-b)
cv.imshow('webcam',frame)

cv.waitKey()
for i in range(len(puntos)):
    print(puntos[i])
    cv.circle(frame,puntos[i],radius=2,thickness=-1,color=(255,0,0))
if len(puntos) > 2:
    v = puntos[1] - puntos[0]
    a = la.norm(v)
    v = v/a
    b = la.norm(puntos[1]-puntos[2])
    cinf = punto_fuga(a,b)
    last = puntos[2]
    print(a,b,v,last)
    for i in range(0,n):
        c = siguiente_punto(a,b)
        last = (last + c*v).astype(np.uint32)
        cv.circle(frame,last,radius=2,thickness=-1,color=(0,255,0))
        a = b
        b = c
    print(cinf)
    pf = (puntos[2] + cinf*v).astype(np.uint32)
    cv.circle(frame,pf,radius=2,thickness=-1,color=(0,0,255))
cv.imshow('webcam',frame)
cv.waitKey(0)
cv.destroyAllWindows()