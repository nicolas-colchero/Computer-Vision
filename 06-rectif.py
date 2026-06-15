#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream

# Idea: Seleccionamos 4 puntos de referencia, conocemos las distancias entre los 4 puntos.
# Formamos una imagen del cuadrilátero en las distancias reales (escaladas)
# Encontramos la homografía que pasa de una imagen a la otra y calculamos la imagen por la homografía de los puntos cuya distancia queremos medir
# Medimos la distancia de las imágenes por la homografía y multiplicamos por el factor de escala


# convierte un conjunto de puntos ordinarios (almacenados como filas de la matriz de entrada)
# en coordenas homogéneas (añadimos una columna de 1)
def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

# convierte en coordenadas tradicionales
def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]
"""
Obtenemos la homografía que transforma de nuestro carnet ideal al carnet en la imagen
Aplicamos dicha homografía a los puntos de los que queremos medir la distancia para obtener los puntos en el mundo real
Medimos la distancia entre los puntos
"""
def calcular_distancia(puntos,puntos_referencia,distancias_reales_escaladas,factor_escala):
    H,_ = cv.findHomography(puntos_referencia,distancias_reales_escaladas)
    puntos_reales = inhomog(homog(puntos) @ H.T)
    distancia = np.linalg.norm(puntos_reales[0] - puntos_reales[1])
    return distancia * factor_escala

def manejador(event, x, y, flags, param):
    global puntos
    global puntos_referencia
    global state
    if event == cv.EVENT_LBUTTONDOWN:
        print(len(puntos_referencia))
        if 'p' == state and len(puntos) < 2:
            puntos = puntos + [[x,y]]
            if 2 == len(puntos): puntos = np.array(puntos)
        if 'r' == state and len(puntos_referencia) < 4:
            puntos_referencia = puntos_referencia + [[x,y]]
            print(puntos_referencia)
            if 4 == len(puntos_referencia): puntos_referencia = np.array(puntos_referencia)

cv.namedWindow('Solution',cv.WINDOW_AUTOSIZE)
cv.setMouseCallback("Solution", manejador)
puntos = []
puntos_referencia = []
distancias_reales_escaladas = np.array([[0,0],[0,108],[171,108],[171,0]])
factor_escala = 0.5
dist = 0
state = 0
for key,frame in autoStream():
    if ord('x') == key:
        state == 0
        puntos = []
        puntos_referencia = []
        dist = 0
    if ord('r') == key and 0 == state:
        state = 'r'
        print('selecciona los puntos de referencia')
    if ord('p') == key and 4 == len(puntos_referencia):
        state = 'p'
        puntos = []
        print('Selecciona los puntos cuya distancia quieres medir')
    if ord('c') == key and 2 == len(puntos):
        dist = calcular_distancia(puntos,puntos_referencia,distancias_reales_escaladas,factor_escala)
    
    if len(puntos)==2: cv.line(frame,puntos[1],puntos[0],(255,0,0))
    for i in range(1,len(puntos_referencia)):
        cv.line(frame,puntos_referencia[i-1],puntos_referencia[i],(0,255,0))
    if len(puntos_referencia)==4: cv.line(frame,puntos_referencia[3],puntos_referencia[0],(0,255,0))

    if dist != 0 and 2 == len(puntos):
        str_loc = (puntos[0] + puntos[1]) // 2
        cv.putText(frame, str(dist) + ' mm', str_loc,cv.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 1)
    cv.imshow('Solution',frame)
cv.destroyAllWindows()