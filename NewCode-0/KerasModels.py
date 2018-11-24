# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# %% ------------------------- Model Version 01 -------------------------


def get_model_v1(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='valid',
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    return model


# %% ------------------------- Model Version 02 -------------------------

def get_model_v2(input_shape, nb_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    return model


# %% ------------------------- Model Version 03 -------------------------

def get_model_v3(input_shape, nb_classes):
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    return model


# %% ------------------------- Model Version 04 -------------------------

def get_model_v4(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid',
                     input_shape=input_shape, activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    return model


# %% ------------------------- Model Version 05 -------------------------

def get_model_v5(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(256, (3, 3), padding='valid',
                     input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))

    return model
