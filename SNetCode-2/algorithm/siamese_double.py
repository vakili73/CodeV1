
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
    model.add(layers.Dense(128, activation='sigmoid', name="dense_128_sigmoid"))
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_b = model(input_b)
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model


def get_model_v2(input_shape, nb_classes):
    model = get_base_v2(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='sigmoid', name="dense_128_sigmoid"))
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_b = model(input_b)
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model


def get_model_v3(input_shape, nb_classes):
    model = get_base_v3(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='sigmoid', name="dense_128_sigmoid"))
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_b = model(input_b)
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model


def get_model_v4(input_shape, nb_classes):
    model = get_base_v4(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='sigmoid', name="dense_128_sigmoid"))
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_b = model(input_b)
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model


def get_model_v5(input_shape, nb_classes):
    model = get_base_v5(input_shape)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='sigmoid', name="dense_128_sigmoid"))
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    processed_a = model(input_a)
    processed_b = model(input_b)
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
# %% train functions


def fit(algorithm, transformed_db, max_iter=1000001, patience=10):
    X_train, y_train = transformed_db.get_train()
    X_test, y_test = transformed_db.get_test()
    nb_classes = transformed_db.nb_classes
    history = {'loss': [],
               'val_loss': []}
    epoch = []
    for _ in range(max_iter):
        in_1, in_2, out_train = get_batch(X_train, y_train, nb_classes)
        loss = algorithm.model.train_on_batch([in_1, in_2], out_train)
        history['loss'].append(loss)
        in_1, in_2, out_test = get_batch(X_test, y_test, nb_classes)
        val_loss = algorithm.model.test_on_batch([in_1, in_2], out_test)
        history['val_loss'].append(val_loss)
        epoch.append(_)
        if not _ % 100:
            train_acc = compute_accuracy(
                algorithm.model.predict([in_1, in_2]), out_train)
            print("Batch %d --> loss: %.5f - val_loss: %.5f - train_acc: %.5f" %
                  (_, loss, val_loss, train_acc))
            if _ and history['val_loss'][-101] <= val_loss:
                patience -= 1
                if not patience:
                    break
    in_1_train, in_2_train, out_train = get_batch(X_train, y_train, nb_classes, batch_size=128*100)
    in_1_test, in_2_test, out_test = get_batch(X_test, y_test, nb_classes, batch_size=128*100)
    transformed_db.set_data([in_1_train, in_2_train], out_train,
                            [in_1_test, in_2_test], out_test)
    return History(epoch, history)


def get_batch(X_train, y_train, nb_classes, batch_size=128):
    train = []
    indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    min_len = min([indices[i].size for i in range(nb_classes)])
    for _ in range(batch_size//2):
        classes_order = np.random.permutation(range(nb_classes))
        anchor = classes_order[0]
        other = np.random.choice(classes_order[1:])
        o_index = np.random.randint(min_len)
        a_index = np.random.randint(min_len)
        train.append((X_train[indices[other][o_index]],
                      X_train[indices[anchor][a_index]], 1))
        a_index = np.random.randint(min_len, size=2)
        train.append((X_train[indices[anchor][a_index[0]]],
                      X_train[indices[anchor][a_index[1]]], 0))
    in_1, in_2, out = zip(*train)
    return np.stack(in_1), np.stack(in_2), np.stack(out)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() > 0.5].mean()


class History:
    def __init__(self, epoch, history):
        self.epoch = epoch
        self.history = history
