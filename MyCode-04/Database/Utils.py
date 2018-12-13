from .Config import CVNAME
from .Config import DBPATH

import os
import _pickle
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.utils import shuffle
from matplotlib import pyplot as plt


# %% Utils function

def load_data(db_name) -> tuple:
    if os.path.exists(DBPATH+'/'+db_name+'/pcache.cpkl'):
        fileObj = open(DBPATH+'/'+db_name+'/pcache.cpkl', 'rb')
        X_train, X_test, y_train, y_test = _pickle.load(fileObj)
        return X_train, X_test, y_train, y_test
    X_train = np.array([], dtype='float')
    y_train = np.array([], dtype='int')
    for cv in CVNAME:
        data = pd.read_csv(DBPATH+'/'+db_name+'/' +
                           cv+'.txt', header=None).values
        if cv == 'E':
            X_test = data[:, 1:].astype('float')
            y_test = data[:, 0].astype('int')
        else:
            X_train = np.concatenate((X_train, data[:, 1:].astype(
                'float')), axis=0) if X_train.size else data[:, 1:].astype('float')
            y_train = np.concatenate((y_train, data[:, 0].astype(
                'int')), axis=0) if y_train.size else data[:, 0].astype('int')
    with open(DBPATH+'/'+db_name+'/pcache.cpkl', 'wb') as fileObj:
        _pickle.dump((X_train, X_test, y_train, y_test), fileObj, protocol=4)
    return X_train, X_test, y_train, y_test


def get_fewshot(X, y, shot=None, way=-1) -> tuple:
    if shot == None:
        return X, y
    way = len(np.unique(y)) if way == -1 else way
    _X = []
    _y = []
    for i in range(way):
        ind = np.where(y == i)[0]
        _X.extend(X[ind[0:shot]])
        _y.extend(y[ind[0:shot]])
    _X = np.array(_X)
    _y = np.array(_y)
    _X, _y = shuffle(_X, _y)
    return _X, _y


def reshape(X, shape) -> tuple:
    img_rows = shape[0]
    img_cols = shape[1]
    channels = shape[2]
    X = X.reshape(
        X.shape[0], img_rows, img_cols, channels)
    return X


def plot_histogram(y, title, save=True, figsize=(19.20, 10.80),
                   base_path='./logs/datahists') -> plt.Figure:
    unique, counts = np.unique(y, return_counts=True)
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title)
    y = counts - np.mean(counts)
    sns.barplot(x=unique, y=y, palette="deep", ax=ax1)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel("Diverging")
    sns.barplot(x=unique, y=counts, palette="deep", ax=ax2)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel("Qualitative")
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'.png'
        fig.savefig(path)
    return fig
