# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

import math
import pickle
import os
import copy
import random
import numpy as np
import pandas as pd

import KerasModels
import keras.layers

from keras import backend as K
from keras.models import Model

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# %% Loading Datasets

def load_ABCDE_datasets(path):
    CVNAMES = ['A', 'B', 'C', 'D', 'E']
    X_train = y_train = np.array([])
    for cv in CVNAMES:
        data = pd.read_csv(path + '/' + cv + '.txt').values
        if cv == 'E':
            X_test = data[:, 1:].astype('float32')
            y_test = data[:, 0].astype('int')
        else:
            X_train = np.concatenate((X_train, data[:, 1:].astype(
                'float32')), axis=0) if X_train.size else data[:, 1:].astype('float32')
            y_train = np.concatenate((y_train, data[:, 0].astype(
                'int')), axis=0) if y_train.size else data[:, 0].astype('int')

    train = {}
    for data_version in [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]:
        train.update({data_version: []})
        for cv in ['A', 'B', 'C', 'D']:
            with open(path + '/' + data_version + '/' + cv + '.pkl', 'rb') as file:
                batch = list(pickle.load(file))
                if cv == 'A': 
                    train[data_version] = batch
                else:
                    for i in range(len(batch)):
                        train[data_version][i] = np.concatenate((train[data_version][i],
                                                                 batch[i]))
                        
    return train, X_train, y_train, X_test, y_test


# %% Data Preprocessing

def preprocess_data(mode, data):
    if mode == None or mode == 'None':
        return data
    elif mode == '255' or mode == 255:
        data /= 255
    elif mode == 'standard':
        data = StandardScaler().fit(data)
    elif mode == 'minmax':
        data = MinMaxScaler().fit(data)
    elif mode == 'mean':
        data -= np.mean(data)
    elif mode == 'std':
        data -= np.std(data)
    else:
        print('\nError: Wrong Preprocessing Mode')
        quit()
    return data


def reshape(X_train, X_test, flat, channels, img_rows=None, img_cols=None):
    if (img_rows == None) and (img_cols == None):
        img_rows = img_cols = int(math.sqrt(X_train.shape[1] / channels))
    else:
        print('\nError: Wrong Reshape Inputs')
        quit()
    if flat == True:
        input_shape = X_train.shape[1]
    elif K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(
            X_train.shape[0], channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        X_train = X_train.reshape(
            X_train.shape[0], img_rows, img_cols, channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
    return X_train, X_test, input_shape


def single_reshape(X_train, flat, channels, img_rows=None, img_cols=None):
    if (img_rows == None) and (img_cols == None):
        img_rows = img_cols = int(math.sqrt(X_train.shape[1] / channels))
    else:
        print('\nError: Wrong Reshape Inputs')
        quit()
    if flat == True:
        input_shape = X_train.shape[1]
    elif K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(
            X_train.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        X_train = X_train.reshape(
            X_train.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
    return X_train, input_shape


# %% Model Generation

def model_generator(model_schema, schema, nb_classes, input_shape, model_version):
    models = []
    for item in schema:
        model = getattr(KerasModels, 'get_model_v' +
                        str(model_version))(input_shape)
        in_out = {}
        for layer, attr in model_schema[item]["structure"]:
            attr = copy.deepcopy(attr)
            if layer in ["Dropout", "Dense"]:
                if 'units' in attr and attr['units'] == None:
                    attr['units'] = nb_classes
                if 'export' in attr:
                    del attr['export']
                model.add(getattr(keras.layers, layer)(**attr))
            elif layer == "Input":
                if isinstance(input_shape, int):
                    in_mdl = keras.layers.Input(shape=(input_shape,))
                elif isinstance(input_shape, tuple):
                    in_mdl = keras.layers.Input(shape=input_shape)
                in_out.update({attr["name"]+'_in': in_mdl})
                in_out.update({attr["name"]+'_out': model(in_mdl)})
            elif layer == "Concatenate":
                merged_vector = getattr(keras.layers, layer)
                inputs = attr['inputs']
                for i in range(len(inputs)):
                    inputs[i] = in_out[inputs[i]]
                merged_vector = merged_vector(axis=attr["axis"], name=attr["name"])(inputs)
                in_out.update({attr["name"]: merged_vector})
            elif layer == "Model":
                inputs = attr['inputs']
                for i in range(len(inputs)):
                    inputs[i] = in_out[inputs[i]]
                outputs = attr['outputs']
                for i in range(len(outputs)):
                    outputs[i] = in_out[outputs[i]]
                model = Model(inputs=inputs, outputs=outputs)
        models.append((item, model))
    return models


# %% Testing and Evaluation

def run_test(name, model, Xdata, Ydata, y_data):
    metrics_names = model.metrics_names
    metrics_scalar = model.evaluate(Xdata, Ydata, verbose=0)

    for metric_name, metric_value in zip(metrics_names, metrics_scalar):
        print("Metric %s" % metric_name)
        print("Metric value: %.5f" % metric_value)

    Ypred = model.predict(Xdata)
    # Report
    print("%s model classification report" % name)
    print(classification_report(
        y_data, np.argmax(Ypred, axis=1), digits=5))


# %% Parallel and Models Saving

def save_model(name, model, path, dataset):
    model.save(path+dataset+"_"+name+".h5")
    model.save_weights(path+dataset+"_"+name+"_weights.h5", overwrite=True)


# %% Feature Extraction

def feature_predict(name, model, model_schema, Xdata):
    layer_name = []
    inter_out = []
    for _, attrib in model_schema[name]["structure"]:
        if "export" in attrib and attrib["export"]:
            layer_name.append(attrib['name'])
            inter_out.append(model.get_layer(attrib['name']).output)
    intermediate_layer_model = Model(model.input, outputs=inter_out)
    intermediate_output = intermediate_layer_model.predict(Xdata)
    return (layer_name, intermediate_output)

def feature_predict_triple(model, structure, Xdata):
    name_layers = []
    inter_layer = []
    for _, attr in structure:
        if "export" in attr and attr["export"]:
            name_layers.append(attr["name"])
            inter_layer.append(model.layers[-2].get_layer(attr["name"]).output)
    intermediate_outputs = Model(model.layers[-2].inputs[0], outputs=inter_layer)
    intermediate_outputs = intermediate_outputs.predict(Xdata)
    return list(zip(name_layers, intermediate_outputs))


# %% CSV Saving

def csv_save(dataset, y_tarin, intermediate_output_tarin,
             y_test, intermediate_output_test, feature_dir):
    train_concat = np.concatenate(
        (y_tarin.reshape((y_tarin.size, 1)), intermediate_output_tarin[1]), axis=1)
    test_concat = np.concatenate(
        (y_test.reshape((y_test.size, 1)), intermediate_output_test[1]), axis=1)

    np.savetxt(feature_dir+dataset + '_' +
               intermediate_output_tarin[0][0]+'_train.txt', train_concat, delimiter=",")
    np.savetxt(feature_dir+dataset + '_' +
               intermediate_output_test[0][0]+'_test.txt', test_concat, delimiter=",")

def csv_save_triple(name, dataset, y_tarin, intermediate_output_tarin,
             y_test, intermediate_output_test, feature_dir):
    for layer, intermediate_output in intermediate_output_tarin:
        train_concat = np.concatenate(
            (y_tarin.reshape((y_tarin.size, 1)), intermediate_output), axis=1)
        np.savetxt(feature_dir+dataset+'_'+layer+'_'+
                name+'_triple_train.txt', train_concat, delimiter=",")

    for layer, intermediate_output in intermediate_output_test:
        test_concat = np.concatenate(
            (y_test.reshape((y_test.size, 1)), intermediate_output), axis=1)
        np.savetxt(feature_dir+dataset+'_'+layer+'_'+
                name+'_triple_test.txt', test_concat, delimiter=",")

# %% Positive and negative pair creation

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)