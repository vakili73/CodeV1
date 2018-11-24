# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

import math
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
    return X_train, y_train, X_test, y_test

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

# %% Model Generation


def model_generator(model_schema, schema, nb_classes, input_shape, model_version):
    models = []
    for item in schema:
        model = getattr(KerasModels, 'get_model_v' +
                        str(model_version))(input_shape, nb_classes)
        for layer, attr in model_schema[item]:
            attr = attr.copy()
            if 'units' in attr and attr['units'] == None:
                attr['units'] = nb_classes
            if 'export' in attr:
                del attr['export']
            model.add(getattr(keras.layers, layer)(**attr))
        models.append((item, model))
    inputs = []
    outputs = []
    for _, model in models:
        inputs.append(model.input)
        outputs.append(model.output)
    parallel_model = Model(inputs=inputs, outputs=outputs)
    return models, parallel_model

# %% Parallel Data Generator


def parallel_data_generator(index, X, Y, no_parallel):
    Xdata = []
    Ydata = []
    for _ in range(no_parallel):
        Xdata.append(X[index, :])
        Ydata.append(Y[index, :])
    return Xdata, Ydata


def parallel_data_generator_for_test(X_train, Y_train, X_test, Y_test, no_parallel):
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for _ in range(no_parallel):
        Xtrain.append(X_train)
        Ytrain.append(Y_train)
        Xtest.append(X_test)
        Ytest.append(Y_test)
    return Xtrain, Ytrain, Xtest, Ytest

# %% Parallel and Models Saving


def save_model(models, parallel_model, path, dataset):
    for name, model in models:
        model.save(path+dataset+"_"+name+".h5")
    parallel_model.save_weights(path+dataset+"_parallel.h5", overwrite=True)

# %% Testing and Evaluation


def run_test(models, parallel_model, Xdata, Ydata, y_data):
    metrics_names = parallel_model.metrics_names
    metrics_scalar = parallel_model.evaluate(Xdata, Ydata, verbose=0)

    for metric_name, metric_value in zip(metrics_names, metrics_scalar):
        print("Metric %s" % metric_name)
        print("Metric value: %.5f" % metric_value)

    Ypred = parallel_model.predict(Xdata)
    # Report
    for i in range(len(models)):
        print("%s model classification report" % models[i][0])
        print(classification_report(
            y_data, np.argmax(Ypred[i], axis=1), digits=5))

# %% Feature Extraction


def feature_predict(models, model_schema, Xdata):
    layer_name = []
    inputs = []
    inter_out = []
    for name, model in models:
        inputs.append(model.input)
        for _, attrib in model_schema[name]:
            if "export" in attrib and attrib["export"]:
                layer_name.append(attrib['name'])
                inter_out.append(model.get_layer(attrib['name']).output)
    intermediate_layer_model = Model(inputs, outputs=inter_out)
    intermediate_output = intermediate_layer_model.predict(Xdata)
    return list(zip(layer_name, intermediate_output))

# %% CSV Saving


def csv_save(dataset, y_tarin, intermediate_output_tarin,
             y_test, intermediate_output_test, models, feature_dir):
    for i in range(len(models)):
        train_concat = np.concatenate(
            (y_tarin.reshape((y_tarin.size, 1)), intermediate_output_tarin[i][1]), axis=1)
        test_concat = np.concatenate(
            (y_test.reshape((y_test.size, 1)), intermediate_output_test[i][1]), axis=1)

        np.savetxt(feature_dir+dataset + '_' +
                   intermediate_output_tarin[i][0]+'_train.txt', train_concat, delimiter=",")
        np.savetxt(feature_dir+dataset + '_' +
                   intermediate_output_test[i][0]+'_test.txt', test_concat, delimiter=",")
