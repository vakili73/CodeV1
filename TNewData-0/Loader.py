import numpy as np
from Database import Dataset

CVNAMES = ['A', 'B', 'C', 'D', 'E']
DATABASE = './Database'

# %% Dataset loader function


def getDataset(name: str = '') -> Dataset:
    if name == '': 
        return Dataset()
    X_train = y_train = np.array([])
    for cv in CVNAMES:
        data = np.genfromtxt(DATABASE+'/'+name+'/'+cv+'.txt', delimiter=',')
        if cv == 'E':
            X_test = data[:, 1:].astype('float64')
            y_test = data[:, 0].astype('int')
        else:
            X_train = np.concatenate((X_train, data[:, 1:].astype(
                'float64')), axis=0) if X_train.size else data[:, 1:].astype('float64')
            y_train = np.concatenate((y_train, data[:, 0].astype(
                'int')), axis=0) if y_train.size else data[:, 0].astype('int')
    return Dataset(name, X_train, y_train, X_test, y_test)


# %% testing
if __name__ == '__main__':
    dataset = getDataset('')
    dataset = getDataset('mnist')
    pass