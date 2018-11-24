# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 18:45:13 2018

@author: vvaki
"""

import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

# %% Loading Dataset


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


X_train, y_train, _, y_test = load_ABCDE_datasets('./')


# %% Calculate Useful Values

classes = np.sort(np.unique(np.concatenate((y_train, y_test))))

classes_index = []
for class_id in classes:
    classes_index.append(np.flatnonzero(y_train == class_id))

num_of_examples = []
for class_indexes in classes_index:
    num_of_examples.append(class_indexes.size)


# %% Triple Data Generator

def choose_random_sample(classes, except_to, classes_index, X_train):
    tmp_classes = classes.copy()
    tmp_classes = np.delete(tmp_classes, except_to)
    rand_class = np.argmax(
        classes == np.random.choice(tmp_classes, replace=False))
    rand_index = np.random.choice(classes_index[rand_class], replace=False)
    return X_train[rand_index, :]


def triple_data_generator_v1(classes, classes_index, num_of_examples, X_train):
    Xtrain = []
    print("Total Classes of v1 is %d" % len(classes))
    for i in range(len(classes)):
        print("Generating v1 data for class %d..." % classes[i])
        comb = list(combinations(classes_index[i], 2))
        perm = np.random.permutation(comb)

        X_ind = []
        for j in range(num_of_examples[i]):
            # % (Negative, Positive, Anchor)
            ind_perm = np.random.permutation([0, 1])
            X_ind.append((None, perm[j][ind_perm[0]], perm[j][ind_perm[1]]))

        for negative, positive, anchor in X_ind:
            negative = choose_random_sample(classes, i, classes_index, X_train)
            Xtrain.append([negative, X_train[positive, :], X_train[anchor, :]])

    np.random.shuffle(Xtrain)
    Xtrain = np.array(Xtrain)
    negative, positive, anchor = Xtrain[:, 0, :], Xtrain[:, 1, :], Xtrain[:, 2, :]
    return negative, positive, anchor


def triple_data_generator_v2(classes, classes_index, num_of_examples, X_train):
    Xtrain = []
    print("Total Classes of v2 is %d" % len(classes))
    for i in range(len(classes)):
        print("Generating v2 data for class %d..." % classes[i])
        comb = list(combinations(classes_index[i], 2))
        perm = np.random.permutation(comb)

        X_ind = []
        for j in range(num_of_examples[i]):
            # % (Negetive or Positive, Anchor, 1 or 0 -> Similarity)
            ind_perm = np.random.permutation([0, 1])
            if np.random.rand() < 0.5:
                X_ind.append((perm[j][ind_perm[0]], perm[j][ind_perm[1]], 0))
            else:  # if both are similar -> distance between them is 0
                X_ind.append((None, perm[j][ind_perm[0]], 1))

        for neg_or_pos, anchor, similarity in X_ind:
            if similarity == 0:
                Xtrain.append((X_train[neg_or_pos, :], X_train[anchor, :], 0))
            elif similarity == 1:
                neg_or_pos = choose_random_sample(
                    classes, i, classes_index, X_train)
                Xtrain.append((neg_or_pos, X_train[anchor, :], 1))

    np.random.shuffle(Xtrain)
    neg_or_pos, anchor, sim_dist = [], [], []
    for n_o_p, a, sd in Xtrain:
        neg_or_pos.append(n_o_p)
        anchor.append(a)
        sim_dist.append(sd)
    neg_or_pos = np.array(neg_or_pos)
    anchor = np.array(anchor)
    sim_dist = np.array(sim_dist)
    return neg_or_pos, anchor, sim_dist


# %% Saving Data to Files

def save_triple_data_v1(path):
    if not os.path.exists(path):
        os.makedirs(path)

    Xtrain = triple_data_generator_v1(
        classes, classes_index, num_of_examples, X_train)

    alphabet = ['A', 'B', 'C', 'D']
    step = int(Xtrain[0].shape[0]/4)
    print("Saving Started for v1...")
    for i in range(4):
        file = open(path + '/' + alphabet[i] + '.pkl', 'wb')
        batch = (Xtrain[0][i*step:(i+1)*step],
                 Xtrain[1][i*step:(i+1)*step],
                 Xtrain[2][i*step:(i+1)*step])
        pickle.dump(batch, file)
        file.close()


def save_triple_data_v2(path):
    if not os.path.exists(path):
        os.makedirs(path)

    Xtrain = triple_data_generator_v2(
        classes, classes_index, num_of_examples, X_train)

    alphabet = ['A', 'B', 'C', 'D']
    step = int(Xtrain[0].shape[0]/4)
    print("Saving Started for v2...")
    for i in range(4):
        file = open(path + '/' + alphabet[i] + '.pkl', 'wb')
        batch = (Xtrain[0][i*step:(i+1)*step],
                 Xtrain[1][i*step:(i+1)*step],
                 Xtrain[2][i*step:(i+1)*step])
        pickle.dump(batch, file)
        file.close()


# %% Program Running
if __name__ == '__main__':

    import multiprocessing

    try:
#        save_triple_data_v1('./triple_data_v1')
        multiprocessing.Process(target=save_triple_data_v1, args=[
                                './triple_data_v1']).start()
#        save_triple_data_v2('./triple_data_v2')
        multiprocessing.Process(target=save_triple_data_v2, args=[
                                './triple_data_v2']).start()

    except:
        print("Error: unable to start thread")
