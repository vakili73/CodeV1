# -*- coding: utf-8 -*-
'''
Created on Sat Aug 25 22:31:09 2018

@author: vahid vakili-zare
@email : v.vakili@pgs.usb.ac.ir
'''

import time
import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# %% Initialization

TRAIN_SIZE = 0
TEST_SIZE = 1
SCORE = 2
PRECISION = 3
RECALL = 4
F1 = 5
TIME = 6
TOTAL_DISTANCES = 7
AVG_DISTANCES = 8


# %% Load dataset fron train / test files

# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

def calculate_clusters(b, X_train, X_test, n_jobs=1):
    kmeans = KMeans(n_clusters=b, random_state=0, n_jobs=n_jobs)
    kmeans.fit(X_train)

    start_time = time.time()
    test_predictions = kmeans.predict(X_test)
    initial_time = float(time.time() - start_time)

    train_clusters = np.zeros((len(X_train), b), dtype=int)
    test_clusters = np.zeros((len(X_test), b), dtype=int)

    for c in range(b):
        train_clusters[:, c] = kmeans.labels_ == c
        test_clusters[:, c] = test_predictions == c

    return train_clusters, test_clusters, initial_time


# -----------------------------------------------------------------------------

def increase_clusters(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, n_jobs=1):
    knn = KNeighborsClassifier(n_neighbors=(k+1), n_jobs=n_jobs)
    knn.fit(X_train, Y_train)
    neighbors_list = knn.kneighbors(X_train, None, False)

    for item_neighbors in neighbors_list:
        item_cluster = np.flatnonzero(train_clusters[item_neighbors[0]])[0]

        for i in range(1, (k+1)):
            neighbour_cluster = np.flatnonzero(
                train_clusters[item_neighbors[i]])[0]

            if neighbour_cluster != item_cluster:
                train_clusters[item_neighbors[i], item_cluster] = 1


# -----------------------------------------------------------------------------

def run_cknn(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, n_jobs=1):
    n_clusters = train_clusters.shape[1]
    total = np.zeros((n_clusters, 9))

    for c in range(n_clusters):
        X_train_cluster = X_train[train_clusters[:, c] == 1]
        Y_train_cluster = Y_train[train_clusters[:, c] == 1]
        X_test_cluster = X_test[test_clusters[:, c] == 1]
        Y_test_cluster = Y_test[test_clusters[:, c] == 1]

        # Non-empty cluster
        if X_train_cluster.shape[0] > 0 and X_test_cluster.shape[0] > 0:
            start_time = time.time()

            k_value = k
            # if the size of the cluster is smaller than k value
            if X_train_cluster.shape[0] < k:
                k_value = X_train_cluster.shape[0]

            knnbc = KNeighborsClassifier(
                n_neighbors=k_value, algorithm='brute', n_jobs=n_jobs)
            knnbc.fit(X_train_cluster, Y_train_cluster)
            Y_pred = knnbc.predict(X_test_cluster)

            knn_time = time.time() - start_time

            # Save results
            score = accuracy_score(Y_test_cluster, Y_pred)
            with warnings.catch_warnings(record=True):  # to ignore warnings
                precision, recall, f1, support = precision_recall_fscore_support(
                    Y_test_cluster, Y_pred, average=None)

            total[c][TRAIN_SIZE] = X_train_cluster.shape[0]
            total[c][TEST_SIZE] = X_test_cluster.shape[0]
            total[c][SCORE] = score
            total[c][PRECISION] = np.average(precision, None, support)
            total[c][RECALL] = np.average(recall, None, support)
            total[c][F1] = np.average(f1, None, support)
            total[c][TIME] = knn_time
            total[c][TOTAL_DISTANCES] = (
                n_clusters * X_test_cluster.shape[0]) + X_train_cluster.shape[0] * X_test_cluster.shape[0]
            total[c][AVG_DISTANCES] = n_clusters + X_train_cluster.shape[0]

    return total

# ----------------------------------------------------------------------------


def print_tabulated(list):
    print('\t'.join('%.4f' % x if type(x) is np.float64 or type(
        x) is float else str(x) for x in list))

# -----------------------------------------------------------------------------


def print_result(dbname, method, cv, b, k, args, X_train, X_test, initial_time, result):
    print_tabulated((dbname, method, 0, 0, (cv+1),
                     X_train.shape[0], X_test.shape[0], b, k,
                     np.average(result[:, SCORE],     None,
                                result[:, TEST_SIZE]),
                     np.average(result[:, PRECISION], None,
                                result[:, TEST_SIZE]),
                     np.average(result[:, RECALL],    None,
                                result[:, TEST_SIZE]),
                     np.average(result[:, F1],        None,
                                result[:, TEST_SIZE]),
                     initial_time + np.sum(result[:, TIME]),
                     np.sum(result[:, TOTAL_DISTANCES]),
                     np.average(result[:, AVG_DISTANCES],
                                None, result[:, TEST_SIZE]),
                     '-', '-', '-',
                     np.std(result[:, SCORE]),
                     np.std(result[:, PRECISION]),
                     np.std(result[:, RECALL]),
                     np.std(result[:, F1]),
                     np.std(result[:, TIME]),
                     '-',
                     np.std(result[:, TRAIN_SIZE])))


# %% Main Program

if __name__ == '__main__':

    # %% Loading Dataset
    path = './feature/'
    all_data = load_all_dataset(path)

    njobs = 8

    for b in (10, 15, 20, 25, 30, 100, 500, 1000):

        print(80*'-')
        print('b=%d' % (b))
        print('\t'.join(('dbname', 'alg.', 'lnoise', 'anoise', 'cv', 'tr_size', 'te_size', 'b', 'k', 'score', 'preci.', 'recall', 'f1',
                         'tmp.sec', 'total_dist', 'm_dist', 'Std:', 'tr_size', 'te_size', 'score', 'preci.', 'recall', 'f1', 'tmp.sec', 't_dist', 'm_dist')))

        for data_name, schemas in all_data.items():
            for schema, data in schemas.items():

                X_train = data['train'][0]
                X_test = data['test'][0]

                Y_train = data['train'][1]
                Y_test = data['test'][1]

                # KMeans
                train_clusters, test_clusters, initial_time = calculate_clusters(
                    b, X_train, X_test, njobs)

                for k in (1, 3, 5, 7, 9):

                    # Run experiment ckNN
                    result_1 = run_cknn(
                        X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, njobs)

                    print_result(data_name+'_'+schema, 'ckNN', 0, b,
                                 k, 0, X_train, X_test, initial_time, result_1)

                    # Run experiment ckNN Plus
                    increase_clusters(
                        X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, njobs)

                    result_2 = run_cknn(
                        X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, njobs)

                    print_result(data_name+'_'+schema, 'ckNN+', 0, b,
                                 k, 0, X_train, X_test, initial_time, result_2)
