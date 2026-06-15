#!/usr/bin/env python
# TODO: Acabar, hay problema con el guardado de vídeos. Idea: Tal vez el problema sea que cogemos x1,y1,x2,y2 en cada iteración. Cogerlo solo una vez
import cv2 as cv
from umucv.stream import autoStream
from umucv.util import Video,ROI
import numpy as np
from collections import deque

"""
Seleccionamos una región.
Se generan 2 vídeos de 3 segundos, uno son los siguientes 3 segundos en esa región, el otro es el detector de movimiento de ese objeto.
El detector de movimiento se hace de 2 formas:
    1 - Utilizando un subtractor de fondo.
    2 - Utilizando frames anteriores y creando un modelo de fondo. Comparamos con el frame actual.
"""

def detector_movimiento_bgsub(img,bgsub,update,kernel):
    fgmask = bgsub.apply(img, learningRate = -1 if update else 0)
    fgmask = cv.erode(fgmask,kernel,iterations = 1)
    fgmask = cv.medianBlur(fgmask,3)
    return fgmask

"""
Suponemos que tenemos un modelo del fondo (imagen de como creemos que es el fondo).
Dada una nueva imagen construimos el nuevo modelo como:
    - modelo = learningRate * frame + (1 - learningRate) * modelo
    - objeto = np.abs(modelo - frame) y aplicamos thresholding
Devuelve el nuevo modelo y una máscara donde esperamos encontrar al objeto
"""
def modelo_fondo(modelo,frame,learningRate = 0.05, update = True):
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if update:
        modelo = learningRate * grey + (1 - learningRate) * modelo
    objeto = np.abs(grey - modelo).astype(np.uint8)
    blur = cv.medianBlur(objeto,5)
    thresh = cv.adaptiveThreshold(blur.astype(np.uint8), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return (modelo, thresh)

cv.namedWindow("input",cv.WINDOW_NORMAL)
cv.namedWindow("Detector",cv.WINDOW_NORMAL)
cv.namedWindow("Region",cv.WINDOW_NORMAL)
cv.moveWindow('input', 0, 0)
roi = ROI("input")
modelo = None
# input = Video(fps=15, codec="MJPG",ext="avi")
movimiento = Video(fps=15, codec="MJPG",ext="avi")

region = []
bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)
use_bgsub = True
update = True
kernel = np.ones((3,3),np.uint8)

for key,frame in autoStream():
    if ord('x') == key:
        modelo = None
        region = []
        # input = Video(fps=15, codec="MJPG",ext="avi")
        movimiento = Video(fps=15, codec="MJPG",ext="avi")
    if ord('b') == key:
        use_bgsub = not use_bgsub
    if ord('u') == key:
        update = not update
    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,0,255), thickness=2)
        region = frame[y1:y2+1,x1:x2+1]
        # input.write(region)
        if use_bgsub:
            detector = detector_movimiento_bgsub(region,bgsub,update,kernel)
        else:
            if modelo is None:
                modelo = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
            (modelo,detector) = modelo_fondo(modelo,region,update=update)
        cv.imshow('Detector',detector)
        cv.imshow('Region',region)
        detector = cv.cvtColor(detector,cv.COLOR_GRAY2BGR)
        movimiento.write(cv.resize(detector,(frame.shape[1],frame.shape[0])),key,ord('v'))
    if ord('r') == key:
        # input.release()
        # input = Video(fps=15, codec="MJPG",ext="avi")
        # input.ON = True
        movimiento.release()
        movimiento = Video(fps=15, codec="MJPG",ext="avi")

    cv.imshow('input',frame)
cv.destroyAllWindows()


