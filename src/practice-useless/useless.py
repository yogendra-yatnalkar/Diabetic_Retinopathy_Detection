# Listing all the imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import imutils
import math

# image height and image width ----> GLOBAL
img_ht = 400
img_wd = 310

train = pd.read_csv('train.csv')
print(train.head())