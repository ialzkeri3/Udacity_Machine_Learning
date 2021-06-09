import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import time
import matplotlib.pyplot as plt
import numpy as np
tfds.disable_progress_bar()
import pandas as pd
import json
import glob

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import argparse 
from PIL import Image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()

parser.add_argument("-i","--image_path", default='./test_images/orange_dahlia.jpg', help = 'The path of the image to make predictions on', type = str)
parser.add_argument("-m","--model", help='path of the trained model', type=str)
parser.add_argument("-k","--top_k",  default = 5, help = 'The number of classes to be shown in the top classes list', type = int)
parser.add_argument("-c","--category_names", default = 'label_map.json', help = 'The path to a JSON file for Mapping categories to real names', type = str)

options = parser.parse_args()

image_path, saved_model, K, class_label_map = options.image_path, options.model, options.top_k, options.category_names

def process_image(test_image):
    processed_img = np.squeeze(test_image)
    processed_img = tf.image.resize(processed_img, (224, 224))
    processed_img /= 255
    
    return processed_img

def predict(image_path, model, class_names, top_k=5):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    
    prediction = model.predict(np.expand_dims(image, 0))
    probs = prediction[0].tolist()
    
    value, index = tf.math.top_k(probs, k=top_k)
    
    probs = value.numpy().tolist()
    labels = index.numpy().tolist()
    
    
    labels[0] = labels[0] + 1
    
    classes = [ class_names[str(i)] for i in labels ]
    
    return probs, classes


if __name__ == "__main__":
    
    model2 = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer':hub.KerasLayer})

    with open(class_label_map, 'r') as f:
        class_names = json.load(f)
    
    probs, classes = predict(image_path, model2, class_names, K)
    
    print('top ' + str(K) + ' probabilities: \n')
    for i in range (K):
        print(str(i+1) + '- ' + classes[i])
        print(probs[i])
        print('\n')