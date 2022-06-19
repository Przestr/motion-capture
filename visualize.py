import pickle
import cv2
import numpy as np
import time

pickle_in = open("gowno.pickle","rb")
exer = pickle.load(pickle_in)

ind = [27, 28, 29, 33, 34, 35, 5, 6, 7, 14, 15, 16, 23]
pairs = [[0,1], [1,2], [7,8], [8,9], [10,11], [11,12], [3,4], [4,5], [0,3], [7,10], [3,10], [0,7] ]

img = np.zeros((1920,1080))

exer = np.array(exer)
exer = exer.tolist()

for i in range(len(exer)): # processing data for proper visualization
    for j in range(len(exer[0])):
        exer[i][j][0]-=500
        exer[i][j][1] = 750 - exer[i][j][1]

for i in range(len(exer)): # loop till the number of frames 
    for j in range(13):  # showw coordinates
        cv2.circle(img, (int(exer[i][j][0]), int(exer[i][j][1])), 1, (255, 255, 255), -10)
    for j in pairs: #  draw skeleton
        cv2.line(img, tuple(exer[i][j[0]]), tuple(exer[i][j[1]]), (255, 255, 255), 1)
    img2 = cv2.resize(img, (1920,1080)) 
    cv2.imshow("img", img2)
    cv2.waitKey(8)

    img = np.zeros((1920,1080))
