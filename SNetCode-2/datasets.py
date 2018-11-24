import os
import numpy as np

# %% Datasets informations


datasets_dir = '/datasets/'

datasets_info = {
    # shape: (--flat, channel, rows, cols)
    "gisette": {
        "origin_model": 3,
        "preprocessing": None,
        "input_shape": (5000,),
        "nb_classes": 2,
        "shape": (True, 1),
    },
    "homus": {
        "origin_model": 5,
        "preprocessing": 255,
        "input_shape": (40, 40, 1),
        "nb_classes": 32,
        "shape": (False, 1),
    },
    "letter": {
        "origin_model": 2,
        "preprocessing": None,
        "input_shape": (16,),
        "nb_classes": 26,
        "shape": (True, 1),
    },
    "mnist": {
        "origin_model": 1,
        "preprocessing": 255,
        "input_shape": (28, 28, 1),
        "nb_classes": 10,
        "shape": (False, 1),
    },
    "nist": {
        "origin_model": 5,
        "preprocessing": 255,
        "input_shape": (32, 32, 1),
        "nb_classes": 26,
        "shape": (False, 1),
    },
    "pendigits": {
        "origin_model": 2,
        "preprocessing": None,
        "input_shape": (16,),
        "nb_classes": 10,
        "shape": (True, 1),
    },
    "satimage": {
        "origin_model": 4,
        "preprocessing": None,
        "input_shape": (3, 3, 4),
        "nb_classes": 6,
        "shape": (False, 4),
    },
    "usps": {
        "origin_model": 1,
        "preprocessing": None,
        "input_shape": (16, 16, 1),
        "nb_classes": 10,
        "shape": (False, 1),
    },
}

# %% Dataset Class


class Dataset:
    def __init__(self, name, structure, base_dir):
        assert isinstance(structure, dict)
        self.name = name
        self.structure = structure
        self.nb_classes = structure['nb_classes']
        self.input_shape = structure['input_shape']
        self.origin_model = structure['origin_model']
        self.preprocessing = structure['preprocessing']
        self.computed_input_shape = None
        self.shape = structure['shape']
        self.base_dir = base_dir

    def set_data(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_path(self):
        path = r'%s' % os.getcwd().replace('\\','/')
        return path + self.base_dir + self.name + '/'

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

# %% Class instantiation function


def get_dataset(dataset):
    assert dataset in datasets_info
    return Dataset(dataset, datasets_info[dataset], datasets_dir)
