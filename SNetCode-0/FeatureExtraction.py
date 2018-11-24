# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 05:08:21 2018

@author: vvaki
"""


import numpy as np
import warnings
import sys

import tensorflow as tf

import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

from sklearn.model_selection import KFold, StratifiedKFold

from LossFunctions import contrastive_loss, triplet_loss

from Util import feature_predict, save_model, single_reshape, create_pairs
from Util import load_ABCDE_datasets, preprocess_data, feature_predict_triple
from Util import reshape, model_generator, run_test, csv_save_triple, csv_save

warnings.filterwarnings('ignore')

# %% Initialization

# config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
# config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
# config.log_device_placement = True

# sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
# set_session(sess)


# %% Parameters Setting

dataset_dir = './dataset/'
feature_dir = './feature/'
model_dir = './model/'

dataset_structure = {
    # shape: [--flat, channel, rows, cols]
    # "gisette_original": {
    #     "model_version": 3,
    #     "preprocessing": None,
    #     "shape": [True, 1],
    #     "schema": ["schema_a", "schema_1", "schema_2"],
    # },
    # "homus_original": {
    #     "model_version": 5,
    #     "preprocessing": 255,
    #     "shape": [False, 1],
    #     "schema": ["schema_d", "schema_1", "schema_2"],
    # },
    # "letter_original": {
    #     "model_version": 2,
    #     "preprocessing": None,
    #     "shape": [True, 1],
    #     "schema": ["schema_b", "schema_1", "schema_2"],
    # },
    # "mnist_original": {
    #     "model_version": 1,
    #     "preprocessing": 255,
    #     "shape": [False, 1],
    #     "schema": ["schema_v2_1", "schema_v2_2"],
    # },
    "mnist_original": {
        "model_version": 6,
        "preprocessing": 255,
        "shape": [False, 1],
        "schema": ["schema_v2_2"],
    },
    # "nist_original": {
    #     "model_version": 5,
    #     "preprocessing": 255,
    #     "shape": [False, 1],
    #     "schema": ["schema_d", "schema_1", "schema_2"],
    # },
    # "pendigits_original": {
    #     "model_version": 2,
    #     "preprocessing": None,
    #     "shape": [True, 1],
    #     "schema": ["schema_b", "schema_1", "schema_2"],
    # },
    # "satimage_original": {
    #     "model_version": 4,
    #     "preprocessing": None,
    #     "shape": [False, 4],
    #     "schema": ["schema_c", "schema_1", "schema_2"],
    # },
    # "usps_original": {
    #     "model_version": 1,
    #     "preprocessing": None,
    #     "shape": [False, 1],
    #     "schema": ["schema_a", "schema_1", "schema_2"],
    # },
}

model_schema = {
    # Proposed schema
    # triple_data_v1 with triplet_loss
    "schema_v1_1": {
        "train_data": "triple_data_v1",
        "loss_func": triplet_loss,
        "metrics": None,
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_v1_1_triple_v1_layer_1", "export": True}),
            ("Dense", {"units": None, "activation": "relu",
                       "name": "schema_v1_1_triple_v1_layer_0", "export": True}),
            *[  # Input and output tensor creation by suffix _in and _out
                ("Input", {"shape": "input_dim", "name": "negative"}),
                ("Input", {"shape": "input_dim", "name": "positive"}),
                ("Input", {"shape": "input_dim", "name": "anchor"}),
            ],
            ("Concatenate", {"inputs": ["negative_out", "positive_out", "anchor_out"],
                             "name": "merged_vector", "axis": -1}),
            ("Model", {"inputs": ["negative_in", "positive_in", "anchor_in"],
                       "outputs": ["merged_vector"]})
        ],
    },
    # triple_data_v2 with contrastive_loss
    "schema_v2_1": {
        "train_data": "triple_data_v2",
        "loss_func": contrastive_loss,
        "metrics": None,
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_v2_1_triple_v2_layer_1", "export": True}),
            ("Dense", {"units": None, "activation": "relu",
                       "name": "schema_v2_1_triple_v2_layer_0", "export": True}),
            *[  # Input and output tensor creation by suffix _in and _out
                ("Input", {"shape": "input_dim", "name": "neg_or_pos"}),
                ("Input", {"shape": "input_dim", "name": "anchor"}),
            ],
            ("Concatenate", {"inputs": ["neg_or_pos_out", "anchor_out"],
                             "name": "merged_vector", "axis": -1}),
            ("Model", {"inputs": ["neg_or_pos_in", "anchor_in"],
                       "outputs": ["merged_vector"]})
        ],
    },
    "schema_v2_2": {
        "train_data": "triple_data_v2",
        "loss_func": contrastive_loss,
        "metrics": None,
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_v2_2_triple_v2_layer_0", "export": True}),
            *[  # Input and output tensor creation by suffix _in and _out
                ("Input", {"shape": "input_dim", "name": "neg_or_pos"}),
                ("Input", {"shape": "input_dim", "name": "anchor"}),
            ],
            ("Concatenate", {"inputs": ["neg_or_pos_out", "anchor_out"],
                             "name": "merged_vector", "axis": -1}),
            ("Model", {"inputs": ["neg_or_pos_in", "anchor_in"],
                       "outputs": ["merged_vector"]})
        ],
    },
    # Original schema
    "schema_a": {
        "train_data": "original",
        "loss_func": "categorical_crossentropy",
        "metrics": ['accuracy'],
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_a_dense_128_relu_layer_1", "export": True}),
            ("Dropout", {"rate": 0.5}),
            ("Dense", {"units": None,
                       "activation": "softmax", "name": "schema_a"}),
        ],
    },
    "schema_b": {
        "train_data": "original",
        "loss_func": "categorical_crossentropy",
        "metrics": ['accuracy'],
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_b_dense_128_relu_layer_1", "export": True}),
            ("Dropout", {"rate": 0.2}),
            ("Dense", {"units": None,
                       "activation": "softmax", "name": "schema_b"}),
        ],
    },
    "schema_c": {
        "train_data": "original",
        "loss_func": "categorical_crossentropy",
        "metrics": ['accuracy'],
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_c_dense_128_relu_layer_1", "export": True}),
            ("Dropout", {"rate": 0.3}),
            ("Dense", {"units": None,
                       "activation": "softmax", "name": "schema_c"}),
        ],
    },
    "schema_d": {
        "train_data": "original",
        "loss_func": "categorical_crossentropy",
        "metrics": ['accuracy'],
        "structure": [
            ("Dense", {"units": 128, "activation": "relu",
                       "name": "schema_d_dense_128_relu_layer_1", "export": True}),
            ("Dropout", {"rate": 0.1}),
            ("Dense", {"units": None,
                       "activation": "softmax", "name": "schema_d"}),
        ],
    },
}


# %% DNN Training

if __name__ == '__main__':

    for dataset in dataset_structure.keys():
        data = {}
        # Loading datasets
        print('Loading Dataset...')
        train, X_train, y_train, X_test, y_test = load_ABCDE_datasets(
            dataset_dir + dataset)
        __TRAIN__ = (X_train.copy(), y_train.copy())
        __TEST__ = (X_test.copy(), y_test.copy())
        data.update({'train': [X_train, y_train], 'test': [X_test, y_test]})
        data.update(train)

        structure = dataset_structure[dataset]
        # loading train and test examples
        print('Data Preprocessing...')
        X_train = preprocess_data(
            structure['preprocessing'], data['train'][0])
        X_test = preprocess_data(
            structure['preprocessing'], data['test'][0])

        y_train = data['train'][1]
        Y_train = to_categorical(y_train)
        y_test = data['test'][1]
        Y_test = to_categorical(y_test)

        # some useful value
        nb_classes = len(np.unique(np.concatenate((y_train, y_test))))
        model_version = structure['model_version']

        # reshaping train and test set
        X_train, X_test, input_shape = reshape(
            X_train, X_test, *structure['shape'])

        print("dataset name: %s" % dataset)
        print("X train shape: %s" % str(X_train.shape))
        print("Y train shape: %s" % str(Y_train.shape))
        print("X test shape: %s" % str(X_test.shape))
        print("Y test shape: %s" % str(Y_test.shape))

        # model generation
        models = model_generator(
            model_schema, structure['schema'], nb_classes, input_shape, model_version)

        for name, model in models:
            print("model name: %s" % name)
            mdl_schema = model_schema[name]
            print("loss: %s" % str(mdl_schema["loss_func"]))
            print("data: %s" % str(mdl_schema["train_data"]))
            model.summary()

            # training phase
            model.compile(loss=mdl_schema["loss_func"], optimizer='adadelta',
                          metrics=mdl_schema["metrics"])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10)

            TRAIN_DATA = mdl_schema["train_data"]
            if TRAIN_DATA == "original":
                kfold = StratifiedKFold(n_splits=5, shuffle=True)
                for train_idx, valid_idx in kfold.split(X_train, y_train):
                    Xtrain, Ytrain = X_train[train_idx], Y_train[train_idx]
                    Xvalid, Yvalid = X_train[valid_idx], Y_train[valid_idx]
                    # fitting parallel model
                    print('Parallel DNN Training...')
                    model.fit(Xtrain, Ytrain, validation_data=(Xvalid, Yvalid),
                              epochs=1000, batch_size=128, verbose=2, callbacks=[early_stopping])

                # model evaluation
                print("Model Name %s" % name)
                print("Training...")
                run_test(name, model, X_train, Y_train, y_train)
                print("Testing...")
                run_test(name, model, X_test, Y_test, y_test)

                # parallel model saving
                print("Model Data and CSV Saving...")
                save_model(name, model, model_dir, dataset)

                # intermediate feature extraction
                print('Intermediate Feature Extracting...')
                intermediate_output_tarin = feature_predict(
                    name, model, model_schema, X_train)
                intermediate_output_test = feature_predict(
                    name, model, model_schema, X_test)

                # save feature as csv
                csv_save(dataset, y_train, intermediate_output_tarin,
                         y_test, intermediate_output_test, feature_dir)

            elif TRAIN_DATA == "triple_data_v1":
                Train = data[TRAIN_DATA]

                print('Data Preprocessing...')
                for i in range(len(Train)):
                    Train[i] = preprocess_data(
                        structure['preprocessing'], Train[i])
                    # reshaping train and test set
                    Train[i], input_shape = single_reshape(
                        Train[i], *structure['shape'])

                kfold = KFold(n_splits=5, shuffle=False)
                for train_idx, valid_idx in kfold.split(Train[-1]):
                    _train = []
                    for i in range(len(Train)):
                        _train.append(Train[i][train_idx])
                    _valid = []
                    for i in range(len(Train)):
                        _valid.append(Train[i][valid_idx])
                    # fitting parallel model
                    print('Parallel DNN Training...')
                    _Y_train = [np.zeros((len(train_idx), model.output.shape.as_list()[-1]))]
                    _Y_valid = [np.zeros((len(valid_idx), model.output.shape.as_list()[-1]))]
                    model.fit(_train, _Y_train, validation_data=(_valid, _Y_valid),
                              epochs=1000, batch_size=128, verbose=2, callbacks=[early_stopping])

                # parallel model saving
                print("Model Data and CSV Saving...")
                save_model(name, model, model_dir, dataset)

                # intermediate feature extraction
                print('Intermediate Feature Extracting...')
                intermediate_outputs_tarin = feature_predict_triple(
                    model, mdl_schema["structure"], X_train)
                intermediate_outputs_test = feature_predict_triple(
                    model, mdl_schema["structure"], X_train)

                # save feature as csv
                csv_save_triple(name, dataset, y_train, intermediate_outputs_tarin,
                         y_test, intermediate_outputs_test, feature_dir)

            elif TRAIN_DATA == "triple_data_v2" and False:
                Train = data[TRAIN_DATA]
                _Xtrain = Train[0:2]
                _Ytrain = Train[2]

                print('Data Preprocessing...')
                for i in range(len(_Xtrain)):
                    _Xtrain[i] = preprocess_data(
                        structure['preprocessing'], _Xtrain[i])
                    # reshaping train and test set
                    _Xtrain[i], input_shape = single_reshape(
                        _Xtrain[i], *structure['shape'])
                
                kfold = KFold(n_splits=5, shuffle=False)
                for train_idx, valid_idx in kfold.split(_Xtrain[-1]):
                    _train = []
                    for i in range(len(_Xtrain)):
                        _train.append(_Xtrain[i][train_idx])
                    _valid = []
                    for i in range(len(_Xtrain)):
                        _valid.append(_Xtrain[i][valid_idx])
                    # fitting parallel model
                    print('Parallel DNN Training...')
                    _Y_train = [np.array(_Ytrain[train_idx])]
                    _Y_valid = [np.array(_Ytrain[valid_idx])]
                    model.fit(_train, _Y_train, validation_data=(_valid, _Y_valid),
                              epochs=1000, batch_size=128, verbose=2, callbacks=[early_stopping])

                # parallel model saving
                print("Model Data and CSV Saving...")
                save_model(name, model, model_dir, dataset)

                # intermediate feature extraction
                print('Intermediate Feature Extracting...')
                intermediate_outputs_tarin = feature_predict_triple(
                    model, mdl_schema["structure"], X_train)
                intermediate_outputs_test = feature_predict_triple(
                    model, mdl_schema["structure"], X_test)

                # save feature as csv
                csv_save_triple(name, dataset, y_train, intermediate_outputs_tarin,
                         y_test, intermediate_outputs_test, feature_dir)

            elif False:
                _Xtrain = __TRAIN__[0]
                _Ytrain = __TRAIN__[1]

                print('Data Preprocessing...')
                _Xtrain = preprocess_data(
                    structure['preprocessing'], _Xtrain)
                # create training+test positive and negative pairs
                digit_indices = [np.where(_Ytrain == i)[0] for i in range(nb_classes)]
                tr_pairs, tr_y = create_pairs(_Xtrain, digit_indices)
                # reshaping train and test set
                tr_pairs_1, input_shape = single_reshape(
                    tr_pairs[:, 0, :], *structure['shape'])
                tr_pairs_2, input_shape = single_reshape(
                    tr_pairs[:, 1, :], *structure['shape'])
                
                kfold = KFold(n_splits=5, shuffle=False)
                for train_idx, valid_idx in kfold.split(tr_pairs):
                    # fitting parallel model
                    print('Parallel DNN Training...')
                    model.fit([tr_pairs_1[train_idx], tr_pairs_2[train_idx]], tr_y[train_idx],
                              validation_data=([tr_pairs_1[valid_idx], tr_pairs_2[valid_idx]], tr_y[valid_idx]),
                              epochs=1000, batch_size=128, verbose=2, callbacks=[early_stopping])

            else:
                from keras.datasets import mnist
                from keras.models import Sequential, Model
                from keras.layers import Dense, Dropout, Input, Lambda
                from keras.optimizers import RMSprop
                # the data, shuffled and split between train and test sets
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                X_train = X_train.reshape(60000, 784)
                X_test = X_test.reshape(10000, 784)
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                input_dim = 784
                nb_epoch = 20

                # create training+test positive and negative pairs
                digit_indices = [np.where(y_train == i)[0] for i in range(10)]
                tr_pairs, tr_y = create_pairs(X_train, digit_indices)

                digit_indices = [np.where(y_test == i)[0] for i in range(10)]
                te_pairs, te_y = create_pairs(X_test, digit_indices)

                base_network = Sequential()
                base_network.add(Dense(128, input_shape=(input_dim,), activation='relu'))
                base_network.add(Dropout(0.1))
                base_network.add(Dense(128, activation='relu'))
                base_network.add(Dropout(0.1))
                base_network.add(Dense(128, activation='relu'))

                input_a = Input(shape=(input_dim,))
                input_b = Input(shape=(input_dim,))

                def euclidean_distance(vects):
                    x, y = vects
                    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

                def eucl_dist_output_shape(shapes):
                    shape1, _ = shapes
                    return (shape1[0], 1)

                # because we re-use the same instance `base_network`,
                # the weights of the network
                # will be shared across the two branches
                processed_a = base_network(input_a)
                processed_b = base_network(input_b)

                distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

                model = Model(input=[input_a, input_b], output=distance)

                def contrastive_loss_(y_true, y_pred):
                    '''Contrastive loss from Hadsell-et-al.'06
                    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
                    '''
                    margin = 1
                    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

                # train
                rms = RMSprop()
                model.compile(loss=contrastive_loss_, optimizer='adadelta')
                
                kfold = KFold(n_splits=5, shuffle=True)
                for train_idx, valid_idx in kfold.split(tr_y):
                    model.fit([tr_pairs[train_idx, 0], tr_pairs[train_idx, 1]], tr_y[train_idx],
                            validation_data=([tr_pairs[valid_idx, 0], tr_pairs[valid_idx, 1]], tr_y[valid_idx]),
                            batch_size=128, nb_epoch=1000, callbacks=[early_stopping])

                def compute_accuracy(predictions, labels):
                    '''Compute classification accuracy with a fixed threshold on distances.
                    '''
                    return labels[predictions.ravel() < 0.5].mean()

                # compute final accuracy on training and test sets
                pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
                tr_acc = compute_accuracy(pred, tr_y)
                pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
                te_acc = compute_accuracy(pred, te_y)

                print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
                print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

