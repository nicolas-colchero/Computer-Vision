#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
import glob

def readbgr(file):
    return cv.resize(cv.imread(file),(500,500))

"""
Calculamos los descriptores SIFT y devolvemos aquellos que sean significativos
"""
def calc_sift(sift, img, minscale = 0):
    kp,desc = sift.detectAndCompute(img,mask = None)
    sc = np.array([k.size for k in kp])
    return desc[sc>minscale].astype(np.uint8)

"""
Dada una imagen x, calcula los descriptores SIFT. Calcula sus etiquetas en el codebook.
Devuelve un histograma de las etiquetas, descartamos equellas que no sean fiables.
"""
def getcode(x):
    desc = calc_sift(sift,x)
    index = codebook.predict(desc)
    r = codebook.cluster_centers_[index] - desc
    d = np.sqrt((r**2).sum(axis=1))
    return np.histogram(index[d<250],np.arange(codebook.n_clusters+1))[0]

def simil(u,v):
    t = max(u.sum(),v.sum())
    return np.minimum(u,v).sum()/t

fcodebook = '../data/codebook.pkl'
fpoints = '../data/keypoints.npz'
fimagecodes = '../data/imagecodes.pkl'

# Creamos el codebook si no existe, si existe lo importamos
sift = cv.xfeatures2d.SIFT_create(nfeatures=500, contrastThreshold = 0.07)
if not os.path.isfile(fpoints):
    imgs = [readbgr(file) for file in glob.glob('../images/SIFT/*')]
    allpoints = []
    for img in imgs:
        allpoints.append(calc_sift(sift,img))
    points = np.vstack(allpoints)
    np.savez_compressed(fpoints, points=points)
else:
    points = np.load(fpoints)['points'].astype(np.float32)
    print(len(points))
if not os.path.isfile(fcodebook):
    codebook = KMeans(n_clusters=300, random_state=0).fit(points)
    joblib.dump(codebook, fcodebook)
else:
    codebook = joblib.load(fcodebook)
if not os.path.isfile(fimagecodes):
    imagecodes = [getcode(x) for x in imgs]
    joblib.dump(imagecodes,fimagecodes)
else:
    imagecodes = joblib.load(fimagecodes)

cv.namedWindow("input",cv.WINDOW_NORMAL)
cv.namedWindow("best",cv.WINDOW_NORMAL)

for key,frame in autoStream():
    frame = cv.resize(frame,(500,500))
    if ord('s') == key:
        filename = '../images/SIFT/image_'+str(len(glob.glob('../images/SIFT/*')))+'.png'
        cv.imwrite(filename,frame)
        img_sift = calc_sift(sift,frame)
        points = np.concatenate([points,img_sift],axis=0)
        os.remove(fpoints)
        np.savez_compressed(fpoints,points=points)
        imagecodes = imagecodes + [getcode(frame)]
        os.remove(fimagecodes)
        joblib.dump(imagecodes,fimagecodes)
    if ord('u') == key:
        os.remove(fcodebook)
        codebook = KMeans(n_clusters=300, random_state=0).fit(points)
        joblib.dump(codebook, fcodebook)
        imagecodes = [getcode(x) for x in imgs]
    hist = getcode(frame)
    dists = sorted([(simil(hist,u),k) for k,u in enumerate(imagecodes)])[::-1]
    best_image_index = dists[0][1]
    best_distance = round(dists[0][0]*100,2)
    second_best_diff = best_distance - round(dists[1][0]*100,2)
    best_image = readbgr(glob.glob('../images/SIFT/*')[best_image_index])
    frame = cv.putText(frame,str(best_distance) + '% + ' + str(second_best_diff)+'%',(10,10),cv.FONT_HERSHEY_SIMPLEX, 0.33, (255,0,0), 1)
    best_image = cv.resize(best_image,(20,20))
    frame[20:40,0:20] = best_image
    cv.imshow('input',frame)
cv.destroyAllWindows()