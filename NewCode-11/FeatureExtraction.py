# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

import warnings
import numpy as np


from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold

from Util import parallel_data_generator_for_test
from Util import preprocess_data, reshape, save_model, csv_save, feature_predict
from Util import load_ABCDE_datasets, parallel_data_generator, run_test, model_generator

warnings.filterwarnings('ignore')

# %% Initialization


dataset_dir = './dataset/'
feature_dir = './feature/'
model_dir = './model/'

dataset_data = {}

model_schema = {
    # Proposed schema
    "schema_1": [
        ("Dense", {"units": 128, "activation": "relu"}),
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_1_dense_none_linear_layer_1", "export": True}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_1"}),
    ],
    "schema_2": [
        ("Dense", {"units": 128, "activation": "relu"}),
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_2_dense_none_linear_layer_2", "export": True}),
        ("Dense", {"units": 128, "activation": "linear"}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_2"}),
    ],
    "schema_3": [
        ("Dense", {"units": 128, "activation": "relu"}),
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_3_dense_none_linear_layer_2", "export": True}),
        ("Dense", {"units": 128, "activation": "linear"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_3"}),
    ],
    "schema_4": [
        ("Dense", {"units": 128, "activation": "relu"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_4_dense_none_linear_layer_1", "export": True}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_4"}),
    ],
    "schema_5": [
        ("Dense", {"units": 128, "activation": "relu"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_5_dense_none_linear_layer_2", "export": True}),
        ("Dense", {"units": 128, "activation": "linear"}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_5"}),
    ],
    "schema_6": [
        ("Dense", {"units": 128, "activation": "relu"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_6_dense_none_linear_layer_2", "export": True}),
        ("Dense", {"units": 128, "activation": "linear"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_6"}),
    ],
    "schema_7": [
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_7_dense_none_linear_layer_1", "export": True}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_7"}),
    ],
    "schema_8": [
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_8_dense_none_linear_layer_2", "export": True}),
        ("Dense", {"units": 128, "activation": "linear"}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_8"}),
    ],
    "schema_9": [
        ("Dense", {"units": 3, "activation": "linear",
                   "name": "schema_9_dense_none_linear_layer_2", "export": True}),
        ("Dense", {"units": 128, "activation": "linear"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 1, "activation": "relu", "name": "schema_9"}),
    ],
}
    
from keras import losses

def  my_loss_plus(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)+0.2, axis=-1)
	
def  my_loss_minus(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)-0.2, axis=-1)

loss_functions = [
        ('my_loss_plus', my_loss_plus), 
        ('my_loss_minus', my_loss_minus), 
        ('mean_squared_logarithmic_error', losses.mean_squared_logarithmic_error), 
        ('squared_hinge', losses.squared_hinge), 
        # ('hinge', losses.hinge), 
        # ('categorical_hinge', losses.categorical_hinge), 
        # ('logcosh', losses.logcosh), 
        # ('categorical_crossentropy', losses.categorical_crossentropy), 
        # ('kullback_leibler_divergence', losses.kullback_leibler_divergence), 
        # ('poisson', losses.poisson), 
        # ('cosine_proximity', losses.cosine_proximity), 
        ]

dataset_structure = {
    # shape: [--flat, channel, rows, cols]
#    "gisette_original": {
#        "model_version": 3,
#        "preprocessing": None,
#        "shape": [True, 1],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
#    "homus_original": {
#        "model_version": 5,
#        "preprocessing": 255,
#        "shape": [False, 1],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
#    "letter_original": {
#        "model_version": 2,
#        "preprocessing": None,
#        "shape": [True, 1],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
    "mnist_original": {
        "model_version": 1,
        "preprocessing": 255,
        "shape": [False, 1],
        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
    },
#    "nist_original": {
#        "model_version": 5,
#        "preprocessing": 255,
#        "shape": [False, 1],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
#    "pendigits_original": {
#        "model_version": 2,
#        "preprocessing": None,
#        "shape": [True, 1],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
#    "satimage_original": {
#        "model_version": 4,
#        "preprocessing": None,
#        "shape": [False, 4],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
#    "usps_original": {
#        "model_version": 1,
#        "preprocessing": None,
#        "shape": [False, 1],
#        "schema": ["schema_1", "schema_2", "schema_3", "schema_4", "schema_5", "schema_6", "schema_7", "schema_8", "schema_9"],
#    },
}

# %% Loading Datasets


print('Loading Dataset...')
for item in dataset_structure.keys():
    X_train, y_train, X_test, y_test = load_ABCDE_datasets(dataset_dir + item)
    dataset_data.update(
        {item: {'train': [X_train, y_train], 'test': [X_test, y_test]}})

# %% DNN Training

for loss_name, loss_func in loss_functions:
    for dataset in dataset_structure.keys():
        data = dataset_data[dataset]
        structure = dataset_structure[dataset]
        # loading train and test examples
        print('Data Preprocessing...')
        X_train = preprocess_data(
            structure['preprocessing'], data['train'][0])
        X_test = preprocess_data(
            structure['preprocessing'], data['test'][0])
    
        y_train = data['train'][1]
        y_test = data['test'][1]
        # maximum = np.max([y_train.max(), y_test.max()])
        # Y_train = y_train/maximum
        # Y_test = y_test/maximum
        # Y_train = to_categorical(y_train)
        # Y_test = to_categorical(y_test)
        Y_train = y_train
        Y_test = y_test
    
        # some useful value
        nb_classes = len(np.unique(y_train))
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
        models, parallel_model = model_generator(
            model_schema, structure['schema'], nb_classes, input_shape, model_version)
    
        print("model generated for %s" % loss_name)
        for name, model in models:
            print("model name: %s" % name)
            model.summary()
            
        # training phase
        parallel_model.compile(loss=loss_func,
                               optimizer='adadelta', metrics=['mae'])
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
    
        for train_idx, valid_idx in kfold.split(X_train, y_train):
            # generate data for parallel consuming
            Xtrain, Ytrain = parallel_data_generator(
                train_idx, X_train, Y_train, len(models))
            Xvalid, Yvalid = parallel_data_generator(
                valid_idx, X_train, Y_train, len(models))
    
            # fitting parallel model
            print('Parallel DNN Training... %s' % loss_name)
            parallel_model.fit(Xtrain, Ytrain, validation_data=(Xvalid, Yvalid),
                               epochs=1000, batch_size=64, verbose=2, callbacks=[early_stopping])
    
        # generate parallel data for test
        Xtrain, Ytrain, Xtest, Ytest = parallel_data_generator_for_test(
            X_train, Y_train, X_test, Y_test, len(models))
    
        # model evaluation
        print("Model Name %s" % name)
        print("Training...")
        run_test(models, parallel_model, Xtrain, Ytrain, y_train)
        print("Testing...")
        run_test(models, parallel_model, Xtest, Ytest, y_test)
    
        # parallel model saving
        print("Model Data and CSV Saving...")
        save_model(models, parallel_model, model_dir, dataset + '_' + loss_name)
    
        # intermediate feature extraction
        print('Intermediate Feature Extracting...')
        intermediate_output_tarin = feature_predict(
            models, model_schema, Xtrain)
        intermediate_output_test = feature_predict(
            models, model_schema, Xtest)
    
        # save feature as csv
        csv_save(dataset + '_' + loss_name, y_train, intermediate_output_tarin,
                 y_test, intermediate_output_test, models, feature_dir)
