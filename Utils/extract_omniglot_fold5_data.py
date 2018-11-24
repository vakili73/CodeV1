import numpy as np

import imageio
import glob

import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold

X = []
y = []

for im_path in glob.glob("./all_images/*.png"):
     im = ~imageio.imread(im_path).astype(np.bool)
     X.append(im)
     y.append(int(im_path[13:-7]))

y = np.array(y, dtype=np.int)-1
print(len(np.unique(y)))

X = np.array(X)
X = np.round(resize(X, (X.shape[0], 28, 28, 1)) * 255.0).astype('uint8')
X = np.reshape(X, (X.shape[0], 28*28*1))

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
