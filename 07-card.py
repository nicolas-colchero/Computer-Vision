#!/usr/bin/env python

import cv2   as cv
from umucv.stream import autoStream
import numpy as np
import imutils

# Idea, utilizamos reconocimiento de bordes para detectar los bordes que existen en la imagen
# Los carnet son redondos en las esquinas -> problema: Solución 1: Aproximamos por polylines y medimos si es un cuadrado
# Solución 2: Mejor - Aplicamos cv.minAreaRect y comparamos áreas.
# Una vez hemos detectado el carnet, podemos conocer en que pixels del carnet se encuentra la imagen, aplicamos la homomorfía que "mete 
# el carnet en la imagen" al vídeo.

# Reconoce el contorno más grande que se parezca a un rectángulo y devuelve el contorno la diferencia relativa debe ser menor que un 5%
def reconocer_rectangulo(img,eps = 0.05):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray,(7,7),3)
    thresh = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    thresh = cv.bitwise_not(thresh)
    cnts = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key = cv.contourArea,reverse = True)
    puzzleCnt = None
    for c in cnts:
        peri = cv.arcLength(c,True)
        approx = cv.approxPolyDP(c,eps*peri,True)
        if len(approx) == 4:
            cv.drawContours(frame,[c],-1,(0,0,255),3)
            puzzleCnt = approx.reshape(4,2)
            break
    return puzzleCnt,frame

"""
Entrada: Imagen, coordenadas de un cuadrilátero en la imagen, medidas del cuadrilátero en el mundo real, 
        lugar dentro del cuadrilátero en el que queremos meter la imagen (en el mundo real, debe ser rectangular y viene dado por sus esquinas superior izquierda e inferior derecha)
        y la imagen que queremos meter.
Salida: Imagen modificada con la imagen metida en el lugar correcto.
Nota: Las medidas del mundo real deberán estar escaladas y deberán ser discretas (pixeles)

Idea: Calculamos la homografía que convierte al cuadrilátero en la imagen al mundo real
"""
def introducir_foto_en_imagen(img, contorno, tam_carnet, pos_foto, img2):
    m,n,_ = img.shape
    supiz,infder = pos_foto
    supiz = np.array(supiz)
    infder = np.array(infder)
    foto = cv.resize(img2,infder - supiz)
    H,_ = cv.findHomography(tam_carnet,contorno)
    auxiliar = np.zeros_like(img)
    auxiliar[supiz[1]:infder[1],supiz[0]:infder[0],:] = foto
    Htrans = cv.warpPerspective(auxiliar, H, (n,m))
    mask = Htrans > 0
    np.copyto(img, Htrans, where=mask)
    return img

tam_carnet = np.array([[0,0],[0,108],[171,108],[171,0]])
pos_foto = [[5,20],[65,90]]
img = cv.imread("./images/percy.jpeg")

for input,frame in autoStream():
    carnet,frame = reconocer_rectangulo(frame)
    if carnet is not None:
        res = introducir_foto_en_imagen(frame,carnet,tam_carnet,pos_foto,img)
        cv.imshow('Resultado',res)
cv.destroyAllWindows()