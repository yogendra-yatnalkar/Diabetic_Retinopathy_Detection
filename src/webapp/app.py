import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import cv2
from PIL import Image
import numpy as np
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
import math 
import os
import sys 
# Some utilites
import numpy as np
from util import *
from image_predict import *


# Declare a flask app
app = Flask(__name__)

model_b3 = load_b3()
model_b5 = load_b5()
model_b5_old = load_b5_old()
model_b5_prc = load_b5_prc()

print('Model loaded.')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        img_arr = np.array(img)
        img_arr = img_arr[:,:,:3]

        ans_b3 = ans_predict(img,model_b3,300)
        print(ans_b3,"ans_b3")
        ans_b5 = ans_predict(img,model_b5,456)
        print(ans_b5,"ans_b5")
        ans_b5_old = ans_predict(img,model_b5_old,456)
        print(ans_b5_old,"ans_b5_old")
        ans_b5_prc = ans_predict_prc(img_arr,model_b5_prc,380)
        print(ans_b5_prc)

        l = [ans_b3,ans_b5,ans_b5_old,ans_b5_prc]
        # l = [ans_b3]
        ans_mode = mode_ans(l)

        return jsonify(result=str(ans_mode))

    return None

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="33", threaded=False)