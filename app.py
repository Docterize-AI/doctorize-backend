from flask import Flask
# 
import numpy as np
import pandas as pd

import os

import tensorflow as tf
from tensorflow import keras

import string
import nltk
import warnings
warnings.filterwarnings("ignore")
import os
import shutil

from tensorflow.keras import layers
from keras.callbacks import EarlyStopping

import pickle
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,array_to_img

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
model  = tf.keras.models.load_model('./CNN_model_3.h5')

dict = {0: 'COVID-19 PNEUMONIA', 
        1: 'NORMAL', 
        2: 'NON-COVID PNEUMONIA'}

localImages = [
   
    'covid2.jpeg',
    'normal1.jpeg',
    'normal2.jpeg',
    'pneumonia1.jpeg',
    'pneumonia2.jpeg',
     'covid1.jpeg',
]
preds = []
predictedClass = []

for e in localImages:
    path = './images/' + e
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    images = np.vstack([x])
    preds = model.predict(images, batch_size = 32)
    predictedClass = np.argmax(preds, axis=1)
    predictedClass = np.vectorize(dict.get)(predictedClass)
    print(type(preds[0]))
# 
app = Flask(__name__)

@app.route("/")
def hello():
  return "result: " + ''.join(predictedClass) + " " + str(np.max(preds))

if __name__ == "__main__":
  app.run()