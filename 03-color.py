#!/usr/bin/env python

import numpy as np
import cv2 as cv
from umucv.stream import autoStream
import imutils

"""
Se calcula la media y varianza de los colores y se buscan los objetos que se encuentran en el rango (mean - sigma * var, mean + sigma * var)
Sigma es un número real, mean y var tienen tantas dimensiones como canales tiene la imagen.
A continuación, aplicamos un filtro gaussiano a la máscara obtenido mediante umbralización y buscamos los bordes. El número de objetos será el número de bordes.
Devuelve los contornos de los objetos, el número de objetos y una máscara de los objetos
"""
def encontrar_objetos(img,colores,sigma=4,minlen=100):
    mean = np.mean(colores,axis=0)
    var = np.sqrt( np.var(colores,axis=0) )
    rng = [mean - sigma * var, mean + sigma * var]
    print(rng)
    mask = cv.inRange(img, rng[0], rng[1] )
    blur = cv.GaussianBlur(mask,(5,5),0)
    (_, gt) = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours = cv.findContours(gt.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    ok = [c.reshape(len(c),2) for c in contours if cv.arcLength(c,closed=True) >= minlen]
    cv.imshow('blur',blur)
    result = np.zeros_like(img)
    cv.drawContours(result, ok, contourIdx=-1, color=(0,255,255), thickness=1, lineType=cv.LINE_AA)
    return ([ c.reshape(-1,2) for c in ok ],len(ok), result)

def gaussian(x, mu, sig):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

"""
Este detector de objetos funciona de forma distinta. Lo que hace es calcular para cada pixel el grado de pertenencia al objeto, suponiendo 
que el grado de pertenencia viene dado por una gaussiana. No nos permite contar el número de objetos, para esto habría que umbralizar esta imagen.
En el caso de que la imagen esté a color, se calcula la media de grado de pertenencia de los 3 canales, para que el resultado sea una imagen en blanco y negro.
"""
def encontrar_objetos_gaussiana(img,colores):
    mean = np.mean(colores,axis=0)
    var = np.sqrt( np.var(colores,axis=0) )
    gaus = gaussian(x=img,mu=mean,sig=var)
    gaus = np.min(gaus,axis=2)
    gaus = gaus / np.amax(gaus)
    return gaus

def manejador(event, x, y, flags, param):
    global state
    global colores
    if event == cv.EVENT_LBUTTONDOWN and 's' == state:
        colores = colores + [frame[y][x]]


cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)
use_gaussian = False
state = None
colores = []

for key,frame in autoStream():
    if ord('g') == key:
        use_gaussian = not use_gaussian
    if ord('s') == key:
        state = 's'
    if ord('c') == key:
        state = 'c'
    if ord('x') == key:
        colores = []
        state = None
        use_gaussian = False
    if 'c' == state and len(colores) > 1:
        print(use_gaussian,colores)
        if use_gaussian:
            gaussian_mask = encontrar_objetos_gaussiana(frame,colores)
            cv.imshow('gaussian mask',gaussian_mask)
        else:
            (_,objetos,result) = encontrar_objetos(frame,colores)
            str_loc = (10,10)
            cv.putText(result, 'Se han encontrado ' + str(objetos) + ' objetos.', str_loc,cv.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 1)
            cv.imshow('resultado',result)
    cv.imshow('webcam',frame)
cv.destroyAllWindows()
