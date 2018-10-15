# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

import os
import time, re
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# %% Load dataset fron train / test files


def load_all_dataset(path):
    dataset = {}
    for root, _, files in os.walk(path):
        for file in files:
            ind = file.find('_schema')
            data_name = file[:ind-1]
            fnd = re.findall('_t[A-Za-z].*', file[ind+1:-4])[0]
            schema_name = file[ind+1:file.find(fnd)]
            if data_name not in dataset:
                dataset.update({data_name: {}})
            if schema_name not in dataset[data_name]:
                dataset[data_name].update({schema_name: {}})
            if file.find('train') != -1:
                print("train: %s" % file)
                train = pd.read_csv(root+file, sep=',').values
                X_train = train[:, 1:].astype('float32')
                y_train = train[:, 0].astype('int')
                dataset[data_name][schema_name].update(
                    {'train': (X_train, y_train)})
            elif file.find('test') != -1:
                print("test: %s" % file)
                test = pd.read_csv(root+file, sep=',').values
                X_test = test[:, 1:].astype('float32')
                y_test = test[:, 0].astype('int')
                dataset[data_name][schema_name].update(
                    {'test': (X_test, y_test)})

    return dataset


def print_tabulated(list):
    print('\t'.join('%.4f' % x if type(x) is np.float64 or type(
        x) is float else str(x) for x in list))


# %% Main Program


if __name__ == '__main__':

    # %% Initialization

    path = './feature/'
    all_data = load_all_dataset(path)

    print(80*'-')
    print('\t'.join(('dbname', 'lnoise', 'anoise', 'l2', 'cv', 'tr_size', 'te_size',
                     'k', 'score', 'preci.', 'recall', 'f1', 'tmp.sec', 'total_dist', 'm_dist')))

    for k in [1, 3, 5, 7, 9]:

        for data_name, schemas in all_data.items():
            for schema, data in schemas.items():
                # Run experiment
                start_time = time.time()

                clf = KNeighborsClassifier(n_neighbors=k, n_jobs=8)
                clf.fit(data['train'][0], data['train'][1])
                Y_pred = clf.predict(data['test'][0])

                total_time = time.time() - start_time

                # Report results
                score = accuracy_score(data['test'][1], Y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    data['test'][1], Y_pred, average=None)

                print_tabulated((data_name+'_'+schema, 0, 0, 'no', 0,
                                 data['train'][0].shape[0], data['test'][0].shape[0], k,
                                 score,
                                 np.average(precision),
                                 np.average(recall),
                                 np.average(f1),
                                 total_time,
                                 data['train'][0].shape[0] *
                                 data['test'][0].shape[0],
                                 data['train'][0].shape[0]))
