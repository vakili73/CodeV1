# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

from tensorflow.keras import layers, Model, Sequential

# %% ------------------------- Model Version 01 -------------------------


def get_base_v1(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='valid',
                            input_shape=input_shape, activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    return model


# %% ------------------------- Model Version 02 -------------------------

def get_base_v2(input_shape):
    model = Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    return model


# %% ------------------------- Model Version 03 -------------------------

def get_base_v3(input_shape):
    model = Sequential()
    model.add(layers.Dense(4096, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    return model


# %% ------------------------- Model Version 04 -------------------------

def get_base_v4(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(64, (1, 1), padding='valid',
                            input_shape=input_shape, activation='relu'))
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (2, 2), padding='valid', activation='relu'))
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


# %% ------------------------- Model Version 05 -------------------------

def get_base_v5(input_shape):
    model = Sequential()
    model.add(layers.Conv2D(256, (3, 3), padding='valid',
                            input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))
    return model
