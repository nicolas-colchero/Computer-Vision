#! /usr/bin/env python

# Crear un mosaico a partir de imágenes en una carpeta a través de homografías (¿y keypoints?). Comparar con stitching de OpenCV
# Podemos encontrar toda la información necesaria en transf2D.iynb
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

def readrgb(file):
    return cv.cvtColor( cv.imread(file), cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)
def desp(d):                                            # Construimos la matriz de desplazamiento
    dx,dy=d
    return np.array([
        [1,0,dx],
        [0,1,dy],
        [0,0,1]])

# Generalizamos el ejercicio
# Calcula la homografía para juntar la imagen query con model, calcula también el número de puntos
# que coinciden en ambas imagenes
def match(query,model):
    x1 = query
    x2 = model
    (k1,d1) = sift.detectAndCompute(x1,None)
    (k2,d2) = sift.detectAndCompute(x2,None)
    matches = bf.knnMatch(d1,d2,k=2)
    good = []
    for m in matches:
        if len(m) == 2:
            best,second = m
            if best.distance < 0.75*second.distance:
                good.append(best)
    if len(good) < 6: return 0, None

    src_pts = np.array([ k2[m.trainIdx].pt for m in good]).astype(np.float32).reshape(-1,2)
    dst_pts = np.array([ k1[m.queryIdx].pt for m in good]).astype(np.float32).reshape(-1,2)

    H,mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC,3)
    return sum(mask.flatten()>0),H

pano = [readrgb(x) for x in sorted(glob.glob('./images/pano/pano*.jpg'))]
sift = cv.SIFT_create()
bf = cv.BFMatcher()
nimgs = len(pano)
coincidencias = sorted([(match(p,q)[0],i,j) for i,p in enumerate(pano) for j,q in enumerate(pano) if i < j],reverse=True)
# For loop, en cada iteración añadimos una imagen
# Si hemos añadidos las imagenes [i1 i2 .. in] filtramos coincidencias tal que
#       coincidencias[1] xor coincidencias[2] == ij
#       Calculamos el maximo (que estará arriba)
incluidas = [coincidencias[0][1]]
h,w,_ = pano[coincidencias[0][1]].shape
mw,mh = 2000,500                                                 # Debería seleccionarse automaticamente
T = desp((mw,float(mh)))
sz = (w+2*mw,h+2*mh)
resultado = cv.warpPerspective(pano[coincidencias[0][1]],T,sz)

for iteracion in range(nimgs-1):
    coincIncl = [(value,i,j) for value,i,j in coincidencias if (i in incluidas) != (j in incluidas)]
    if len(coincIncl)==0:
        print('Algo va mal')
        break
    if(coincIncl[0][1] in incluidas): 
        nuevaIdx = coincIncl[0][2]
        nueva = pano[coincIncl[0][2]]
    else:
        nuevaIdx = coincIncl[0][1]
        nueva = pano[coincIncl[0][1]]

    nueva = cv.warpPerspective(nueva,T,sz)
    _,HRN = match(resultado,nueva)
    
    cv.warpPerspective(nueva,HRN,sz,nueva,0,cv.BORDER_TRANSPARENT)
    resultado = np.maximum(resultado,nueva)
    incluidas.append(nuevaIdx)
resultado = cv.cvtColor( resultado, cv.COLOR_RGB2BGR) 
cv.imshow('res',resultado)
resultado = cv.cvtColor( resultado, cv.COLOR_BGR2RGB) 
cv.waitKey(0)


(k1,d1) = sift.detectAndCompute(resultado,None)
x2 = readrgb('./images/pano004.jpg')
x2 = cv.warpPerspective(x2,T,sz)
(k2,d2) = sift.detectAndCompute(x2,None)
matches = bf.knnMatch(d2,d1,k=2)
good = []
for m in matches:
    if len(m) == 2:
        best,second = m
        if best.distance < 0.75*second.distance:
            good.append(best)

src_pts = np.array([ k1[m.trainIdx].pt for m in good]).astype(np.float32).reshape(-1,2)
dst_pts = np.array([ k2[m.queryIdx].pt for m in good]).astype(np.float32).reshape(-1,2)

H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3) # cv.LMEDS
matchesMask = mask.ravel()>0
ok = [ good[k] for k in range(len(good)) if matchesMask[k] ]

img = cv.drawMatches(x2,k2,resultado,k1,ok,flags=2,outImg=None,matchColor=(128,0,0))
img = cv.cvtColor( img, cv.COLOR_RGB2BGR) 
cv.imshow('img',img)
cv.waitKey(0)
print(mask.shape)