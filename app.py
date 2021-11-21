from flask import Flask, redirect, url_for, request
from flask_cors import CORS, cross_origin
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

import librosa
import librosa.display

app = Flask(__name__)
# CORS(app, support_credentials=True)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello():
  return "hello"

app.config["IMAGE_UPLOADS"] = "./images"
app.config["AUDIO_UPLOADS"] = './audio'

model  = tf.keras.models.load_model('./CNN_model_3.h5')  

dict = {
  0: 'COVID-19 Pneumonia', 
  1: 'Normal', 
  2: 'Non-COVID Pneumonia',
  3: 'Tuberculosis'
}

#To use the doctorize button in our web-app
@app.route('/upload',methods=['POST'])
@cross_origin(supports_credentials=True)
def upload():
  print(request.form['fileType'])
  if request.form['fileType'] == 'image':
    uploadedImgStr = request.files["file"]
    filepath = os.path.join(app.config["IMAGE_UPLOADS"], uploadedImgStr.filename)
    uploadedImgStr.save(filepath)
    print("stored as: "+filepath)

    img = image.load_img(filepath, target_size=(224,224))
    # print(img)
    x = image.img_to_array(img)
    # print(x)
    x = np.expand_dims(x, axis = 0)
    model=''

    # print(request.data)
    # print(request.args)
    print(request.form)

    if request.form["imgType"] == 'brain':
      model = tf.keras.models.load_model('models/CNN_model_brain.h5')
      dict = {
        0: 'Glioma Tumor', 
        1: 'Meningioma Tumor', 
        2: 'no Tumor',
        3: 'Pituitary Tumor'
      }
    elif request.form["imgType"] == 'skin':
      model = tf.keras.models.load_model('models/CNN_model_skin.h5')
      dict = {
        0: 'Benign', 
        1: 'Malignant'
      }
    elif request.form["imgType"] == 'xray':
      model  = tf.keras.models.load_model('./CNN_model_3.h5')  
      dict = {
        0: 'COVID-19 Pneumonia', 
        1: 'no disease', 
        2: 'Non-COVID Pneumonia',
        3: 'Tuberculosis'
      }
    else:
      return "Image type doesn't match the uploaded photo. Plase choose the proper image type."

    images = np.vstack([x])
    preds = model.predict(images, batch_size = 32)
    predictedClass = np.argmax(preds, axis=1)
    predictedClass = np.vectorize(dict.get)(predictedClass)
    # "result: " + ''.join(predictedClass) + " " + str(np.max(preds))
    return ''.join(predictedClass) + "," + str(np.max(preds))
  else:
    model2  = tf.keras.models.load_model('models/cough.h5')
    dict = {
      0: 'NOT COVID', 
      1: 'COVID'
    }
    uploadedAudStr = request.files["file"]
    filepath = os.path.join(app.config["AUDIO_UPLOADS"], uploadedAudStr.filename)
    uploadedAudStr.save(filepath)
    print("stored as: "+filepath)
    y,sr = librosa.load(filepath, mono=True, duration=5)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    df_row = list(np.array([np.mean(chroma_stft),np.mean(rmse), np.mean(spec_cent),  np.mean(spec_bw), np.mean(rolloff) , np.mean(zcr)])) + list(np.array([[np.mean(e)] for e in mfcc]).flatten())
    arr = np.asarray(df_row)

    preds = model2.predict(arr.reshape(1,26))
    predictedClass = np.argmax(preds, axis=1)
    predictedClass = np.vectorize(dict.get)(predictedClass)
    print(predictedClass)
    return ''.join(predictedClass) + "," + str(np.max(preds))

if __name__ == "__main__":
  app.run()