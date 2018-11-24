
# pylint: disable=W0614
import numpy as np
from .base import *  # noqa
from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

def get_model_v1(input_shape, nb_classes):
    model = get_base_v1(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu', name="dense_128_relu"))
    input_a = layers.Input(shape=input_shape)
    input_p = layers.Input(shape=input_shape)
    input_n = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_p = model(input_p)
    processed_n = model(input_n)
    concatenate = layers.concatenate([processed_a, processed_p, processed_n])
    model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
    return model


def get_model_v2(input_shape, nb_classes):
    model = get_base_v2(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu', name="dense_128_relu"))
    input_a = layers.Input(shape=input_shape)
    input_p = layers.Input(shape=input_shape)
    input_n = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_p = model(input_p)
    processed_n = model(input_n)
    concatenate = layers.concatenate([processed_a, processed_p, processed_n])
    model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
    return model


def get_model_v3(input_shape, nb_classes):
    model = get_base_v3(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu', name="dense_128_relu"))
    input_a = layers.Input(shape=input_shape)
    input_p = layers.Input(shape=input_shape)
    input_n = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_p = model(input_p)
    processed_n = model(input_n)
    concatenate = layers.concatenate([processed_a, processed_p, processed_n])
    model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
    return model


def get_model_v4(input_shape, nb_classes):
    model = get_base_v4(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu', name="dense_128_relu"))
    input_a = layers.Input(shape=input_shape)
    input_p = layers.Input(shape=input_shape)
    input_n = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_p = model(input_p)
    processed_n = model(input_n)
    concatenate = layers.concatenate([processed_a, processed_p, processed_n])
    model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
    return model


def get_model_v5(input_shape, nb_classes):
    model = get_base_v5(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu', name="dense_128_relu"))
    input_a = layers.Input(shape=input_shape)
    input_p = layers.Input(shape=input_shape)
    input_n = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_p = model(input_p)
    processed_n = model(input_n)
    concatenate = layers.concatenate([processed_a, processed_p, processed_n])
    model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
    return model

# %% train functions


def fit(algorithm, transformed_db, max_iter=1000001, patience=10):
    X_train, y_train = transformed_db.get_train()
    X_test, y_test = transformed_db.get_test()
    nb_classes = transformed_db.nb_classes
    history = {'loss': [],
               'val_loss': []}
    epoch = []
    for _ in range(max_iter):
        anchor, positive, negetive = get_batch(X_train, y_train, nb_classes)
        loss = algorithm.model.train_on_batch([anchor, positive, negetive], anchor)
        history['loss'].append(loss)
        anchor, positive, negetive = get_batch(X_test, y_test, nb_classes)
        val_loss = algorithm.model.test_on_batch([anchor, positive, negetive], anchor)
        history['val_loss'].append(val_loss)
        epoch.append(_)
        if not _ % 100:
            print("Batch %d --> loss: %.5f - val_loss: %.5f" %
                  (_, loss, val_loss))
            if _ and history['val_loss'][-101] <= val_loss:
                patience -= 1
                if not patience:
                    break
    anchor_tr, positive_tr, negetive_tr = get_batch(X_train, y_train, nb_classes, batch_size=128*100)
    anchor_te, positive_te, negetive_te = get_batch(X_test, y_test, nb_classes, batch_size=128*100)
    transformed_db.set_data([anchor_tr, positive_tr, negetive_tr], anchor_tr,
                            [anchor_te, positive_te, negetive_te], anchor_te)
    return History(epoch, history)


def get_batch(X_train, y_train, nb_classes, batch_size=128):
    train = []
    indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    min_len = min([indices[i].size for i in range(nb_classes)])
    for _ in range(batch_size):
        classes_order = np.random.permutation(range(nb_classes))
        anchor = classes_order[0]
        positive = classes_order[0]
        negetive = np.random.choice(classes_order[1:])
        a_index = np.random.randint(min_len)
        p_index = np.random.randint(min_len)
        n_index = np.random.randint(min_len)
        train.append((X_train[indices[anchor][a_index]],
                      X_train[indices[positive][p_index]],
                      X_train[indices[negetive][n_index]]))
    anchor, positive, negetive = zip(*train)
    return np.stack(anchor), np.stack(positive), np.stack(negetive)


class History:
    def __init__(self, epoch, history):
        self.epoch = epoch
        self.history = history
