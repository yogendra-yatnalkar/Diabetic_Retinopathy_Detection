import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import imutils
import math

img1 = cv2.imread('ans.png')
img2 = cv2.imread('anscr.png')
img1 = imutils.resize(img1, height = 520, width = 400) 
img2 = imutils.resize(img2, height = 520, width = 400)    

h,w,c=img1.shape
imgf=np.zeros((h,w,c),dtype=np.uint8)
cv2.imshow('display_name', imgf)

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv',mask_inv)

imgf = cv2.bitwise_and(img2,img1)
cv2.imshow('final', imgf)

cv2.waitKey(0)
cv2.destroyAllWindows()