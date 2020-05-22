# image height and image width ----> GLOBAL

import cv2
from PIL import Image
import numpy as np
import os
import math
import sys
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import DenseNet121
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import random
# Some utilites
import numpy as np
from util import base64_to_pil


img_ht = 380
img_wd = 380

def displayImage(display_name, image):
    cv2.namedWindow(display_name,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(display_name, image)

def findContourEye(thresh_image):
    cnts = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
    cnts = max(cnts[0], key=cv2.contourArea)
    return cnts

def findContourEyeExtreme(cnts):
    # Locating extreme points on all 4 sides
    leftmost = tuple(cnts[cnts[:,:,0].argmin()][0])
    rightmost = tuple(cnts[cnts[:,:,0].argmax()][0])
    topmost = tuple(cnts[cnts[:,:,1].argmin()][0])
    bottommost = tuple(cnts[cnts[:,:,1].argmax()][0])
    # Locating the top left and bottom right corner
    x1 = leftmost[0]
    y1 = topmost[1]
    x2 = rightmost[0]
    y2 = bottommost[1]
    return x1,y1,x2,y2 

def findRadiusAndCentreOfContourEye(cnts):
    M = cv2.moments(cnts)
    if( M["m00"]==0):
        cX, cY = 0, 0
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    if(cX < cY):
        r = cX
    else:
        r = cY
    return cX,cY,r

def drawCentreOnContourEye(image,cnts,cX,cY):
    cv2.drawContours(image, [cnts], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20),
    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    return image
    
def Radius_Reduction(img,cX,cY,r):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(cX),int(cY)),int(r), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1

def imageResize(image, ht, wd):
    # resized_image = imutils.resize(image, height = ht, width = wd)
    resized_image = cv2.resize(image,(wd,ht))
    return resized_image

def crop_black(image):
    org = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
    # displayImage('thresh',thresh)
    cnts = findContourEye(thresh)
    x1,y1,x2,y2 = findContourEyeExtreme(cnts)
#     print(x1,y1,x2,y2)
#     crop = org[y1:y2, x1:x2]
#     crop = imageResize(crop, img_ht, img_wd)
#     # displayImage("cr1",crop)
#     return crop
    ext_x = int((x2 - x1)*4//100)
    ext_y = int((y2 - y1)*5//100)
#     print(ext_x,ext_y)
    crop = org[y1+ext_y:y2-ext_y, x1+ext_x:x2-ext_x]
    crop = imageResize(crop, img_ht, img_wd)
    return crop

def imageAugmentation(image):
    x_flip = cv2.flip( image, 0 )
    y_flip = cv2.flip( image, 1 )
    xy_flip = cv2.flip(x_flip,1)
    return x_flip, y_flip, xy_flip

def imageHistEqualization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    sub_med = cv2.addWeighted (im, 1, bg, -1, 255)
    return sub_med

def colorEnhancement(image1,image2):
    image_final = cv2.bitwise_and(image1,image2)
    return image_final

def imageAugSave(path,img1,img2,img3,img4,img_ht,img_wd):
    count = len(os.listdir(path))

    img1 = imageResize(img1, img_ht, img_wd)
    img2 = imageResize(img2, img_ht, img_wd)
    img3 = imageResize(img3, img_ht, img_wd)
    img4 = imageResize(img4, img_ht, img_wd)

    cv2.imwrite(os.path.join(path , '%d.png'%(count+1)), img1)
    cv2.imwrite(os.path.join(path , '%d.png'%(count+2)), img2)
    cv2.imwrite(os.path.join(path , '%d.png'%(count+3)), img3)
    cv2.imwrite(os.path.join(path , '%d.png'%(count+4)), img4)
    return count+1,count+2,count+3,count+4

def processed_test_save(path,img,img_ht,img_wd):
    count = len(os.listdir(path))
    img = imageResize(img,img_ht,img_wd)
    cv2.imwrite(os.path.join(path , '%d.png'%(count+1)), img)
    return count+1

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def new_preprocess_image(image, desired_size=380):
    # image = cv2.imread(image_path)
    # image = preprocess_image(image_path,desired_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = imageResize(image, desired_size, desired_size)
    print(image.shape)
    image_crop = crop_black(image)
    image_clahe = imageHistEqualization(image_crop)
    sub_med = subtract_median_bg_image(image_clahe)
    image_final = colorEnhancement(sub_med, image_clahe)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_final


def preprocess_image(im, desired_size=380):
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    return im

def build_model(cnn_net):
    model = Sequential()
    model.add(cnn_net)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model

import efficientnet.tfkeras as efn

def load_b4():
    cnn_net = efn.EfficientNetB4(weights='imagenet',include_top=False,input_shape=(380, 380, 3))
    model = build_model(cnn_net)
    model.load_weights('models/effnet_b4.h5')
    return model

def load_b3():
    cnn_net = efn.EfficientNetB3(weights='imagenet',include_top=False,input_shape=(300, 300, 3))
    model = build_model(cnn_net)
    model.load_weights('models/effnet_b3.h5')
    return model

def load_b5():
    cnn_net = efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(456, 456, 3))
    model = build_model(cnn_net)
    model.load_weights('models/effnet_b5.h5')
    return model

def load_b5_old():
    cnn_net = efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(456, 456, 3))
    model = build_model(cnn_net)
    model.load_weights('models/effnet_b5_old_new.h5')
    return model

def load_b5_prc():
    cnn_net = efn.EfficientNetB5(weights='imagenet',include_top=False,input_shape=(380, 380, 3))
    model = build_model(cnn_net)
    model.load_weights('models/effnet_b5_old_new_preprocess.h5')
    return model

# model_b4 = load_b4()

def ans_predict(img,model,desired_size):
    img = preprocess_image(img,desired_size)
    img = np.expand_dims(img,axis = 0)
    ans = model.predict(img) > 0.5
    ans = (ans.astype(int).sum(axis=1) - 1)[0]
    return ans

def ans_predict_prc(img,model,desired_size):
    img = new_preprocess_image(img,desired_size)
    img = np.expand_dims(img,axis = 0)
    ans = model.predict(img) > 0.5
    ans = (ans.astype(int).sum(axis=1) - 1)[0]
    return ans

def mode_ans(lst):
    l_ans = []
    d_mem_count = Counter(lst)
    cnt_max = max(d_mem_count.values())
    for k in d_mem_count.keys():
        if(d_mem_count[k] == cnt_max):
            l_ans.append(k)
    if(len(l_ans) == 1):
        return "Diabetic Retinopathy class is :"+str(l_ans[0])
    else:
        l_ans.sort()
        return "Diabetic Retinopathy class is between class : "+str(l_ans[0])+" to class : "+str(l_ans[-1])
# im = Image.open('1.png')
# ans = ans_predict(im,model_b4,380)
# print(ans)

# def image_predict(img):

