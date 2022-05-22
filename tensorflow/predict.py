import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

from scipy.stats import rankdata
from PIL import Image

import argparse


def process_image(test_image):
    img = tf.cast(test_image, tf.float32)
    img /= 255
    resized_image = tf.image.resize(img, [224, 224])

    return resized_image.numpy()


def make_prediction(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)
    processed_image = np.expand_dims(processed_test_image, axis=0)

    ps = model.predict(processed_image)
    ranks = rankdata([-1 * i for i in ps[0]]).astype(int)

    indices = []
    probs = []

    for k in range(1, top_k + 1):
        for i in range(len(ranks)):
            if ranks[i] == k:
                indices.append(i)
                probs.append(ps[0][i])

    # Add one because class labels start from 1
    labels = np.array(indices) + 1

    classes = []
    for label in labels:
        classes.append(class_names[str(label)])

    return probs, classes


parser = argparse.ArgumentParser(description='Classify a flower image.')
parser.add_argument('image_path', help='path to image file')
parser.add_argument('model', help='saved keras h5 model')
parser.add_argument('--top_k', type=int, default=1, help='top k classes ranked by predicted probability')
parser.add_argument('--category_names', default='label_map.json', help='label map json file')
args = parser.parse_args()


with open(args.category_names, 'r') as f:
    class_names = json.load(f)

reloaded_keras_model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
probs, classes = make_prediction(args.image_path, reloaded_keras_model, args.top_k)

for i in range(len(probs)):
    print('Class name: {}'.format(classes[i]), ', Probability: {}'.format(probs[i]))
