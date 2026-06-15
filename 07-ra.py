#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# En esta versión el objeto virtual se mueve.

# pruébalo con el vídeo de siempre

# ./pose2.py --dev=../../images/rot4.mjpg

# con la imagen de prueba

# ./pose2.py --dev=dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import cube, showAxes
from umucv.contours import extractContours, redu


def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])


stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT


K = Kfov( size, 60 )
xMouse = -1
yMouse = -1
tMouse = -1
xCube = 0
yCube = 0
xRealCube=0
yRealCube=0

marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])

square = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [1,   1,   0],
        [1,   0,   0]])



def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]

def manejador(event, x, y, flags, param):
    global xMouse
    global yMouse
    global tMouse
    if event == cv.EVENT_LBUTTONDOWN:
        xMouse = x
        yMouse = y
        tMouse = n

cv.namedWindow('Output')
cv.setMouseCallback("Output", manejador)

for n, (key,frame) in enumerate(stream):

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]
    poses = [poses[2]]
    for M in poses:
        # capturamos el color de un punto cerca del marcador para borrarlo
        # dibujando un cuadrado encima
        x,y = htrans(M, (0.7,0.7,0) ).astype(int)
        if xCube == -1 or yCube == -1:
            xCube = x + 0.75
            yCube = y + 0.75
#        b,g,r = frame[y,x].astype(int)
#        cv.drawContours(frame,[htrans(M,square*1.1+(-0.05,-0.05,0)).astype(int)], -1, (int(b),int(g),int(r)) , -1, cv.LINE_AA)
        # cv.drawContours(frame,[htrans(M,marker).astype(int)], -1, (0,0,0) , 3, cv.LINE_AA)
        
        # Mostramos el sistema de referencia inducido por el marcador (es una utilidad de umucv)
        showAxes(frame, M, scale=0.5)

        # hacemos que se mueva el cubo
        if tMouse != -1:
            xRealCube,yRealCube = htrans(M,(xCube,yCube,0)).astype(int)
            xRealCube,yRealCube = ((xRealCube-xMouse)/frame.shape[1],(yRealCube-yMouse)/frame.shape[0])
            xCube += xRealCube*0.02
            yCube += yRealCube*0.02
            cosa = cube * (0.75,0.75,0.75) + (yCube,xCube,0)
        else:
            cosa = cube * (0.75,0.75,0.75)
        cv.drawContours(frame, [ htrans(M, cosa).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)

    cv.imshow('Output',frame)
    
