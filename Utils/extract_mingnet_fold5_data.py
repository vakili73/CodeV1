import _pickle
import numpy as np

from itertools import repeat

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def _open(path):
    with open(path, 'rb') as fileObj:
        return _pickle.load(fileObj)

train = _open('./mini-imagenet-cache-train.pkl')
test = _open('./mini-imagenet-cache-test.pkl')
val = _open('./mini-imagenet-cache-val.pkl')


X = []
y = []

def _ext(data):
    for _, val in data['class_dict'].items():
        X.append(data['image_data'][val])

_ext(train)
_ext(test)
_ext(val)

for i in range(100):
    times = list(repeat(i, X[i].shape[0]))
    y.extend(times)
y = np.array(y, dtype=np.int)
X = np.concatenate(X, axis=0)
X = np.reshape(X, (X.shape[0], 84*84*3))

concat = np.c_[y, X]

skf = StratifiedKFold(n_splits=5, shuffle=True)

file_name = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt']
file_index = 0

for _, test_index in skf.split(X, y):
    np.savetxt(file_name[file_index], concat[test_index, :], fmt='%d', delimiter=',')
    file_index += 1
    test = y[test_index]
    unique, counts = np.unique(test, return_counts=True)
    print(dict(zip(unique, counts)))