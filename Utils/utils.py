# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers as layer

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def getBaseModel(input_shape, nb_classes):
    img_input = layer.Input(shape=input_shape)
    conv_1 = layer.Conv2D(filters=64, kernel_size=(
        3, 3), strides=(2, 2), padding='valid')(img_input)
    maxp_1 = layer.MaxPooling2D(pool_size=(2, 2))(conv_1)
    btch_1 = layer.BatchNormalization()(maxp_1)
    conv_2 = layer.Conv2D(filters=128, kernel_size=(
        3, 3), strides=(2, 2), padding='valid', activation='relu')(btch_1)
    maxp_2 = layer.MaxPooling2D(pool_size=(2, 2))(conv_2)
    flat_1 = layer.Flatten()(maxp_2)
    dens_1 = layer.Dense(units=512, activation='relu')(flat_1)
    dens_2 = layer.Dense(units=128, activation='relu')(dens_1)
    cat_output = layer.Dense(units=nb_classes, activation='softmax')(dens_2)

    model = keras.Model(inputs=img_input, outputs=cat_output)

    return model
