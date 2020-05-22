# Listing all the imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import imutils
import math

# image_name = input("Enter the name of image to be processed : ")
image = cv2.imread('4.png')
image = imutils.resize(image, height = 520, width = 400)    
# image = cv2.resize(image, (600, 600))    
org = image.copy()

def displayImage(image, display_name):
    cv2.namedWindow(display_name,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(display_name, image)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = max(cnts, key=cv2.contourArea)
# print(cnts)
# print(thresh[cX,cY])
leftmost = tuple(cnts[cnts[:,:,0].argmin()][0])
rightmost = tuple(cnts[cnts[:,:,0].argmax()][0])
topmost = tuple(cnts[cnts[:,:,1].argmin()][0])
bottommost = tuple(cnts[cnts[:,:,1].argmax()][0])
# print('leftmost',leftmost)
# print('rightmost',rightmost)
# print('topmost',topmost)
# print('bottommost',bottommost)

x1 = leftmost[0]
y1 = topmost[1]
x2 = rightmost[0]
y2 = bottommost[1]
ht = int(y1 - y2)
wd = int(x2 - x1)
print(x1,y1,'---',x2,y2)

c = cnts
M = cv2.moments(cnts)
if( M["m00"]==0):
    cX, cY = 0, 0
else:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

print(cX,cY)
if(cX < cY):
    r = cX
else:
    r = cY

# draw the contour and center of the shape on the image
cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
cv2.putText(image, "center", (cX - 20, cY - 20),
    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)


def Radius_Reduction(img,cX,cY,r):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    # cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)
    cv2.circle(Frame,(int(cX),int(cY)),int(r), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1


# crop = org[y1:y2 - y1, x1:x2 - x1]
crop = org[y1:y2, x1:x2]
crop = imutils.resize(crop, height = 520, width = 400)    

#-----CLAHE----------------------------------- 
lab= cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# -------------------------------------------

def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 100)

smed = subtract_median_bg_image(final)


img1 = Radius_Reduction(smed,cX,cY,r)
#-----Augmentation-------------------------------
x_flip = cv2.flip( crop, 0 )
y_flip = cv2.flip( crop, 1 )
xy_flip = cv2.flip(x_flip,1)


imgf = cv2.bitwise_and(smed,final)
displayImage(imgf, 'imgf')

# normalizedImg = cv2.normalize(imgf,  normalizedImg, alpha=50, beta=255, norm_type=cv2.NORM_MINMAX)

# show the image
# displayImage(image, 'cnts')
# displayImage(thresh, 'thresh')
displayImage(img1, 'img1')
displayImage(org, 'org')
displayImage(crop, 'crop')
displayImage(final, 'final')
# displayImage(normalizedImg, 'normalizedImg')
displayImage(smed, 'smed')
# cv2.imwrite('ans.png',smed)
# cv2.imwrite('anscr.png',crop)
# displayImage(x_flip, 'x')
# displayImage(y_flip, 'y')
# displayImage(xy_flip, 'xy')




cv2.waitKey(0)
cv2.destroyAllWindows()