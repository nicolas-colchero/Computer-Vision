#!/usr/bin/env python

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
from sudoku import Sudoku
import cv2   as cv
from umucv.stream import autoStream
import numpy as np

from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Softmax, Flatten, GaussianNoise
from tensorflow.keras.utils import img_to_array

# Encuentra errores en el sudoku
def findErrores(puzzle):
    errores = set()
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] is None or puzzle[i][j] == 0: continue
            for i2 in range(i+1,9):
                if puzzle[i][j] == puzzle[i2][j]:
                    errores.add((i,j))
                    errores.add((i2,j))
            for j2 in range(j+1,9):
                if puzzle[i][j] == puzzle[i][j2]:
                    errores.add((i,j))
                    errores.add((i,j2))
            cell = [i//3,j//3]
            position = [i%3, j%3]
            for i2 in range(3):
                for j2 in range(3):
                    if i2 != position[0] and j2 != position[1]:
                        if puzzle[i][j] == puzzle[cell[0]*3+i2][cell[1]*3+j2]:
                            errores.add((i,j))
                            errores.add((cell[0]*3+i2,cell[1]*3+j2))
    return errores

# Resuelve el Sudoku utilizando el paquete Sudoku
def solveSudoku(grid):
    puzzle = Sudoku(3, 3, board=grid.tolist())
    puzzle.show()
    solution = puzzle.solve()
    solution.show_full()
    return solution.board

# Muestra el sudoku en una imagen
def sudoku2img(puzzle, errores,cell):
    imagen = np.zeros((28*9,28*9,3),np.uint8)
    for i in range(1,9):
        if i%3 == 0: thick = 3
        else: thick = 1
        cv.line(imagen,(0,i*28),(9*28,i*28),(255,255,255),thick)
        cv.line(imagen,(i*28,0),(i*28,9*28),(255,255,255),thick)
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] is None: continue
            color = (0,0,255) if (i,j) in errores else (255,255,0)
            cv.putText(imagen, str(puzzle[i][j]), (i*28+10, (j+1)*28-10),cv.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
    if cell is not None:
        cv.line(imagen,(cell[0]*28,cell[1]*28),((cell[0]+1)*28,cell[1]*28),(255,0,255),thick)
        cv.line(imagen,(cell[0]*28,cell[1]*28),(cell[0]*28,(cell[1]+1)*28),(255,0,255),thick)
        cv.line(imagen,((cell[0]+1)*28,(cell[1]+1)*28),((cell[0]+1)*28,cell[1]*28),(255,0,255),thick)
        cv.line(imagen,((cell[0]+1)*28,(cell[1]+1)*28),(cell[0]*28,(cell[1]+1)*28),(255,0,255),thick)
    return imagen

# Dado un fragmento de la imagen del sudoku, lo modificamos para que sea más fácil de clasificar
def extract_digit(cell):
    print(cell.shape)
    _,thresh = cv.threshold(cell,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours , h = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    ok = [ x.reshape(-1,2) for x in contours if len(x)> 50 ]
    if len(ok) == 0: return None
    c = max(ok, key=cv.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # Dibuja los contornos llenos, (thickness < 0)
    cv.drawContours(mask, [c], -1, 255, -1)
    h,w = thresh.shape
    percentFilled = cv.countNonZero(mask) / float(w*h)
    if percentFilled < 0.05: return None
    digit = cv.bitwise_and(thresh,mask)
    return digit

# La imagen debe contener solo al sudoku, de forma que cada casilla se encuentre en [i*w//9:(i+1)*w//9,j*w//9:(j+1)*w//9]
def extractSudoku(imgSudoku,model):
    h,w = imgSudoku.shape
    h9 = h // 9
    w9 = w // 9
    sudoku = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            digit = extract_digit(imgSudoku[i*h9:(i+1)*h9,j*w9:(j+1)*w9].copy())
            if digit is not None:
                roi = cv.resize(digit,(28,28)).astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                sudoku[i][j] = model.predict(roi).argmax(axis=1)[0]
    return sudoku

# Entrena (utilizando MNIST) un modelo de reconocimiento de digitos
def trainModel():
    # Preparamos los datos
    ((trainData,trainLabels),(testData,testLabels)) = mnist.load_data()
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))
    trainData = trainData.astype("float32") / 255.0
    testData = testData.astype("float32") / 255.0

    # Convertimos las etiquetas en vectores binarios
    le = LabelBinarizer()
    trainLabels = le.fit_transform(trainLabels)
    testLabels = le.transform(testLabels)

    # Creamos y entrenamos el modelo
    model = keras.Sequential(
        [
            keras.Input(shape=(28,28,1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    batch_size = 128
    epochs = 15
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(trainData, trainLabels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    return model

def extractContours(img):
    # Extraemos los contornos de la imagen en orden descendiente por el tamaño de área
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray,(7,7),3)
    thresh = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    thresh = cv.bitwise_not(thresh)
    cnts = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key = cv.contourArea,reverse = True)
    # El contorno cuadrado más grande se considerará el sudoku
    puzzleCnt = None
    for c in cnts:
        peri = cv.arcLength(c,True)
        approx = cv.approxPolyDP(c,0.02*peri,True)
        if len(approx) == 4:
            puzzleCnt = approx
            break
    if puzzleCnt is None:
        raise Exception('Could not find Sudoku outline. Try increasing thresholding and contour steps.')
    # Extraemos el soudoku de la imagen. Usamos imutils, ya que ya se ha escrito un código similar en el ejercicio de swap
    puzzle = four_point_transform(img,puzzleCnt.reshape(4,2))
    warped = four_point_transform(gray,puzzleCnt.reshape(4,2))
    return (puzzle,warped)
def findEdges(img):
    # Extraemos las esquinas del sudoku (similar a la función anterior)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray,(7,7),3)
    thresh = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    thresh = cv.bitwise_not(thresh)
    cnts = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key = cv.contourArea,reverse = True)
    # El contorno cuadrado más grande se considerará el sudoku
    puzzleCnt = None
    for c in cnts:
        peri = cv.arcLength(c,True)
        approx = cv.approxPolyDP(c,0.02*peri,True)
        if len(approx) == 4:
            puzzleCnt = approx
            mask = np.zeros_like(img)
            cv.fillPoly(mask, pts=[puzzleCnt.reshape(4,2)], color=(255,255,255))
            mask = mask > 128
            break
    if puzzleCnt is None:
        raise Exception('Could not find Sudoku edges. Try increasing thresholding and contour steps.')
    return (puzzleCnt.reshape(4,2),mask)


def manejador(event, x, y, flags, param):
    global cell
    if event == cv.EVENT_LBUTTONDOWN:
        cell = (x//28,y//28)

cv.namedWindow('Sudoku',cv.WINDOW_AUTOSIZE)
cv.namedWindow('Solution',cv.WINDOW_AUTOSIZE)
cv.setMouseCallback("Sudoku", manejador)
model = trainModel()
sudoku = None
sudokuImg = None
shSudoku = False
solution = None
solutionImg = None
shSolution = False
cell = None
keyboardNumbers = [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7'),ord('8'),ord('9')]
for key,frame in autoStream():
    if key in keyboardNumbers and sudoku is not None and cell is not None:
        number = [(keyNum,i) for keyNum,i in enumerate(keyboardNumbers) if key == keyNum]
        number = number[0][1]
        sudoku[cell] = number
        errores = findErrores(sudoku)
        sodukuImg = sudoku2img(sudoku,errores,cell)
        cv.putText(puzzle, str(sudoku[cell[0]][cell[1]]), (cell[0]*28+10, (cell[1]+1)*28-10),cv.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,0), 1)
    if key == ord('x'):
        sudoku = None
        sudokuImg = None
        solution = None
        solutionImg = None
    if key == ord('t'):
        model = trainModel()
    if key == ord('s') and model is not None:
        if shSudoku:
            shSudoku = False
            continue
        if sudokuImg is not None:
            shSudoku = True
            continue
        (puzzle,warped) = extractContours(frame)
        sudoku = extractSudoku(warped,model)
        errores = findErrores(sudoku)
        sodukuImg = sudoku2img(sudoku,errores,cell)
        shSudoku = True
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] is None: continue
                color = (0,0,255) if (i,j) in errores else (255,255,0)
                cv.putText(puzzle, str(puzzle[i][j]), (i*28+10, (j+1)*28-10),cv.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
    if key == ord('c'):
        if shSolution:
            shSolution = False
            continue
        if solutionImg is not None:
            shSolution = True
            continue
        errores = findErrores(sudoku)
        if len(errores) > 0:
            print('No puede resolverse, hay errores')
            continue
        solution = solveSudoku(sudoku)
        solutionImg = sudoku2img(solution,[],None)
        shSolution = True
    if sudoku is not None:
        m,n,_ = frame.shape
        puntos1 = np.array([[0,0],[0,n],[m,n],[m,0]])
        (puntos2,mask) = findEdges(frame)
        H,_ = cv.findHomography(puntos1, puntos2)
        Htrans = cv.warpPerspective(puzzle,H,(n,m))
        np.copyto(frame,Htrans, where=mask)
    cv.imshow('Input',frame)
    # Falta: poner el sudoku en frame. Estan puestos en warped, falta hacer transformacion de warped a imagen
        # manejador en sudoku para seleccionar región. Una región seleccionada puede modificar el valor del sudoku
    if shSudoku:
        cv.imshow('Sudoku',sodukuImg)
    if shSolution:
        cv.imshow('Solution',solutionImg)
cv.destroyAllWindows()
