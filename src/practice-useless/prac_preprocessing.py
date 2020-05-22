# Listing all the imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import imutils
import math
import prac_pipeline as ppl

# image height and image width ----> GLOBAL
img_ht = 512
img_wd = 512
path_toCollect =  './new_test_2'
path_toSave = './new_train_2'

def appendPNG(img_id):
    return img_id + '.png'

def feedToPipeline(image_name,diagnosis_type):
    global path_toCollect
    global path_toCollect
    global img_ht,img_wd
    global trained_data, train_data

    print(train_data[train_data['id_code']== image_name].index.item(),image_name,diagnosis_type)
    # print(train_data[train_data['id_code']== image_name].index.item())
    
    image = cv2.imread(os.path.join(path_toCollect,image_name))
    image = ppl.imageResize(image, img_ht, img_wd)
    org_copy = image.copy()
    image_crop = ppl.crop_black(image)
    image_clahe = ppl.imageHistEqualization(image_crop)
    sub_med = ppl.subtract_median_bg_image(image_clahe)
    image_final = ppl.colorEnhancement(sub_med, image_clahe)
    aug1, aug2, aug3 = ppl.imageAugmentation(image_final)
    count1,count2,count3,count4 = ppl.imageAugSave(path_toSave,image_final, aug1, aug2, aug3,img_ht,img_wd)
    count1 = str(count1) + '.png'
    count2 = str(count2) + '.png'
    count3 = str(count3) + '.png'
    count4 = str(count4) + '.png'
    len_trained_data = len(trained_data)
    trained_data.loc[len_trained_data]   = [count1,diagnosis_type] 
    trained_data.loc[len_trained_data+1] = [count2,diagnosis_type] 
    trained_data.loc[len_trained_data+2] = [count3,diagnosis_type] 
    trained_data.loc[len_trained_data+3] = [count4,diagnosis_type] 

start = time.time()

train_data = pd.read_csv('train.csv')
newDataframe_cols = ['id_code','diagnosis'] 
trained_data = pd.DataFrame(columns=newDataframe_cols)

# 
# np.vectorize(feedToPipeline)(train_data['id_code'],train_data['diagnosis'])
# 
print(len(train_data))
for i in range(len(train_data)): 
    feedToPipeline(train_data['id_code'][i],train_data['diagnosis'][i])

trained_data.to_csv('final_trained.csv')

end = time.time()
print(end - start)

''' NO USE CODE 

# This code below is used to appnd png to each image_id
train_data['id_code'] = train_data['id_code'].apply(appendPNG)

# This code below was used to remove anomaly in the image_id such as the + sign .

anomaly_id = []
for i in range(len(train_data)):
    image_name_temp = train_data['id_code'][i]
    if('+' in image_name_temp):
        print(i,image_name_temp,train_data['diagnosis'][i])
        anomaly_id.append(image_name_temp)
        train_data = train_data.drop( train_data[train_data['id_code']== image_name_temp].index)
        print('anomaly')
        train_data.to_csv('train.csv')

'''

