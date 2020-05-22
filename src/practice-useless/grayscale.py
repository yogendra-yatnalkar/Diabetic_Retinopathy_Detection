# Write a function to create a grayscale version of an image

#Listing imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import imutils

image_name = input("Enter the name of image to be processed : ")
img = cv2.imread(image_name)

def convertToGray(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image_gray

def convertToChannels(image):
    image_blue, image_green, image_red = cv2.split(image)
    return image_blue, image_green, image_red 

def displayImage(image, display_name):
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
    cv2.imshow(display_name, image)

def edgeDetection(image):
    gray = convertToGray(image)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray,0,30)
    return edged
    
def convertToBinary(image):
    thresh = 10
    ret,thresh = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
    return thresh

img_gray = convertToGray(img)
img_b,img_g, img_r = convertToChannels(img)
displayImage(img_gray, 'Gray Image')
# displayImage(img_b, 'Blue Image')
# displayImage(img_g, 'Green Image')
# displayImage(img_r, 'Red Image')
# displayImage(img, 'Image')
img_thresh = convertToBinary(img)
displayImage(img_thresh, 'Image_thresh')

img_edge = edgeDetection(img_thresh)
displayImage(img_edge, 'Image_edge')

img_merge = cv2.merge((img_b,img_g, img_r))
# displayImage(img_merge, 'Merge Image')

cv2.waitKey(0)
cv2.destroyAllWindows()
