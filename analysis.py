# -*- coding: utf-8 -*-
import os

import tensorflow as tf

classifierLoad = tf.keras.models.load_model('model.h5')
cancer_dir = 'cancer/test/Cancer/'
normal_dir = 'cancer/test/Normal/'
import numpy as np
from keras.preprocessing import image

def analysis():
    y_pred = []
    y_true = []
    for image_ in os.listdir(cancer_dir):
        test_image = image.load_img(cancer_dir + image_, target_size=(200, 200))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        y_true.append(1)
        if result[0][0] == 1:
            y_pred.append(1)
        elif result[0][1] == 1:
            y_pred.append(0)

    for image_ in os.listdir(normal_dir):
        test_image = image.load_img(normal_dir + image_, target_size=(200, 200))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        y_true.append(0)
        if result[0][0] == 1:
            y_pred.append(1)
        elif result[0][1] == 1:
            y_pred.append(0)

    return y_pred,y_true
