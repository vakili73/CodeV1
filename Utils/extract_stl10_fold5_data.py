# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:10:22 2018

@author: vvaki
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)
    
# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# path to the binary train file with image data
TRAIN_DATA_PATH = './data/stl10_binary/train_X.bin'
TEST_DATA_PATH = './data/stl10_binary/test_X.bin'

# path to the binary train file with labels
TRAIN_LABEL_PATH = './data/stl10_binary/train_y.bin'
TEST_LABEL_PATH = './data/stl10_binary/test_y.bin'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images.reshape((-1, 96*96*3))

if __name__ == "__main__":
    # test to check if the whole dataset is read correctly
    train_images = read_all_images(TRAIN_DATA_PATH)
    test_images = read_all_images(TEST_DATA_PATH)
    images = np.concatenate((train_images, test_images))

    train_labels = read_labels(TRAIN_LABEL_PATH)
    test_labels = read_labels(TEST_LABEL_PATH)
    labels = np.concatenate((train_labels, test_labels))
    
    plt.imshow(np.reshape(images[2, :], (96, 96, 3))); plt.show()
    
    concat = np.c_[labels, images]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    file_name = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt']
    file_index = 0
    
    for _, test_index in skf.split(images, labels):
        np.savetxt(file_name[file_index], concat[test_index, :], fmt='%d', delimiter=',')
        file_index += 1
        test = labels[test_index]
        unique, counts = np.unique(test, return_counts=True)
        print(dict(zip(unique, counts)))