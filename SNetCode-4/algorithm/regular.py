# pylint: disable=W0614
from .base import *  # noqa
from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold


def get_model_v1(input_shape, nb_classes):
    model = get_base_v1(input_shape)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_classes, activation='softmax', name="dense_nb_classes_softmax"))
    return model


def get_model_v2(input_shape, nb_classes):
    model = get_base_v2(input_shape)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(nb_classes, activation='softmax', name="dense_nb_classes_softmax"))
    return model


def get_model_v3(input_shape, nb_classes):
    model = get_base_v3(input_shape)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_classes, activation='softmax', name="dense_nb_classes_softmax"))
    return model


def get_model_v4(input_shape, nb_classes):
    model = get_base_v4(input_shape)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(nb_classes, activation='softmax', name="dense_nb_classes_softmax"))
    return model


def get_model_v5(input_shape, nb_classes):
    model = get_base_v5(input_shape)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(nb_classes, activation='softmax', name="dense_nb_classes_softmax"))
    return model


# %% train functions

def fit(algorithm, transformed_db):
    X_train, y_train = transformed_db.get_train()
    Y_train = to_categorical(y_train)
    X_test, y_test = transformed_db.get_test()
    Y_test = to_categorical(y_test)
    history = algorithm.model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=1000, batch_size=128,
                        verbose=2, callbacks=algorithm.callback)
    return history
