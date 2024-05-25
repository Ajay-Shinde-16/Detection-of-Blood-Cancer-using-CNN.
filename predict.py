# -*- coding: utf-8 -*-
import os

import tensorflow as tf

classifierLoad = tf.keras.models.load_model('model.h5',compile=False)
import numpy as np
# from keras.preprocessing import image
import keras.utils as image

def predict(img_):
    test_image = image.load_img(img_, target_size=(200, 200))
    test_image = np.expand_dims(test_image, axis=0)
    result = classifierLoad.predict(test_image)# [[1,0]] or [[0,1]]
    print(result)
    if result[0][0] == 1:
        return 1
    elif result[0][0] == 0:
        return 0


