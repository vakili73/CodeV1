from Config import Estm
from Database import Dataset
from Schema import BaseSchema

import os
import _pickle
import numpy as np
import pandas as pd

CVNAMES = ['A', 'B', 'C', 'D', 'E']
DATABASE = './Database'

# %% Dataset loader function


def getDataset(name: str = '') -> Dataset:
    if name == '':
        return Dataset()
    if os.path.exists(DATABASE+'/'+name+'/Full.cp'):
        fileObj = open(DATABASE+'/'+name+'/Full.cp', 'rb')
        X_train, y_train, X_test, y_test = _pickle.load(fileObj)
        return Dataset(name, X_train, y_train, X_test, y_test)
    X_train = y_train = np.array([])
    for cv in CVNAMES:
        data = pd.read_csv(DATABASE+'/'+name+'/'+cv+'.txt').values
        if cv == 'E':
            X_test = data[:, 1:].astype('float32')
            y_test = data[:, 0].astype('int')
        else:
            X_train = np.concatenate((X_train, data[:, 1:].astype(
                'float32')), axis=0) if X_train.size else data[:, 1:].astype('float32')
            y_train = np.concatenate((y_train, data[:, 0].astype(
                'int')), axis=0) if y_train.size else data[:, 0].astype('int')
    with open(DATABASE+'/'+name+'/Full.cp', 'wb') as fileObj:
        _pickle.dump((X_train, y_train, X_test, y_test), fileObj)
    return Dataset(name, X_train, y_train, X_test, y_test)


def getSchema(db: Dataset, version, estm) -> BaseSchema:
    module = __import__('Schema')
    schema = getattr(module, 'Schema'+version)()
    if estm == Estm.Conventional:
        schema.buildConventional(db.get_shape(), db.info['n_cls'])
    elif estm == Estm.Siamese:
        schema.buildSiamese(db.get_shape(), db.info['n_cls'])
    elif estm == Estm.Triplet:
        schema.buildTriplet(db.get_shape(), db.info['n_cls'])
    return schema


# %% testing
if __name__ == '__main__':
    dataset = getDataset('')
    dataset = getDataset('mnist')

    pass
