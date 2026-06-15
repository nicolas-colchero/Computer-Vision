#!/usr/bin/env python

import cv2 as cv
import imutils
import numpy.linalg as la
import numpy.fft as fft
import numpy as np

def invar(c, wmax=10):
    x,y = c.T
    z = x+y*1j
    f  = fft.fft(z)
    fa = abs(f)                     # para conseguir invarianza a rotación 
                                    # y punto de partida
    s = fa[1] + fa[-1]              # el tamaño global de la figura para normalizar la escala
    v = np.zeros(2*wmax+1)          # espacio para el resultado
    v[:wmax] = fa[2:wmax+2];        # cogemos las componentes de baja frecuencia, positivas
    v[wmax:] = fa[-wmax-1:];        # y negativas. Añadimos también la frecuencia -1, que tiene
                                    # que ver con la "redondez" global de la figura
    
    if fa[-1] > fa[1]:              # normalizamos el sentido de recorrido
        v[:-1] = v[-2::-1]
        v[-1] = fa[1]
    
    return v / s

def mindist(c,mods,labs):
    ds = [(la.norm(c-mods[m]),labs[m]) for m in range(len(mods)) ]
    return sorted(ds, key=lambda x: x[0])

def best_shape(img_bgr,feats,labels,minlen=50):
    img = 255 - cv.cvtColor(img_bgr,cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img,(7,7),3)
    ret3,img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cnts,_ = cv.findContours(img.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cnts = [c.reshape(len(c),2) for c in cnts if cv.arcLength(c,closed=True) >= minlen]
    cnts = sorted(cnts,key=lambda x:x[0,0])
    invars = [invar(c) for c in cnts]
    dists = np.array([mindist(x,feats,labels) for x in invars])
    best_labels = dists[:,0,1]
    dist_to_best = np.array(dists[:,0,0])
    ok_labels = []
    for i in range(len(best_labels)):
        if dist_to_best[i].astype(np.float64) < 0.1 and best_labels[i] != 'I':
            ok_labels.append(best_labels[i])
            cv.drawContours(img_bgr,cnts,i,(0,0,255),3)
            print(dist_to_best[i],best_labels[i])
    return (ok_labels,img_bgr,img)

fmodel = "./images/shapes/platestemplates.jpg"
fimg = "./images/shapes/plate8.jpg"
minlen = 50
labels = "0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ"

model_bgr = cv.imread(fmodel)
model = 255 - cv.cvtColor(model_bgr,cv.COLOR_BGR2GRAY)
model = cv.GaussianBlur(model,(7,7),3)

cv.imshow('model',model)
cv.waitKey(0)

ret3,model = cv.threshold(model,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

cv.imshow('model',model)
cv.waitKey(0)

model_cnts,_ = cv.findContours(model.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
model_cnts = [c.reshape(len(c),2) for c in model_cnts if cv.arcLength(c,closed=True) >= minlen]
model_cnts = sorted(model_cnts,key=lambda x:x[0,0])
feats = [invar(m) for m in model_cnts]
img_bgr = cv.imread(fimg)
cv.imshow('model',model)
cv.waitKey(0)
ok_labels,img_bgr,img = best_shape(img_bgr,feats,labels,minlen=minlen)
print(ok_labels)
cv.imshow('Img_BGR',img_bgr)
cv.waitKey(0)
cv.imshow('Img',img)
cv.waitKey(0)
cv.destroyAllWindows()
