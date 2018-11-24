# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Dataset importer
from keras.utils import np_utils
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist

# Utils
from utils import getBaseModel

print(tf.__version__)

datasets = [
    ("mnist", (28, 28, 1), mnist, 10),
    # ("cifar10", (32, 32, 3), cifar10, 10),
    # ("cifar100", (32, 32, 3), cifar100, 100),
    ("fashion_mnist", (28, 28, 1), fashion_mnist, 10),
]

for dataset_name, input_shape, dataset, nb_classes in datasets:

    # %% loading dataset
    print("%s Dataset with Shape %s" % (dataset_name, str(input_shape)))
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    print("Train Shape: %s, %s" % (str(x_train.shape), str(y_train.shape)))
    print("Test Shape: %s, %s" % (str(x_test.shape), str(y_test.shape)))

    # %% preprocessing
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = np.reshape(x_train, (-1,)+input_shape)
    x_test = np.reshape(x_test, (-1,)+input_shape)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # %% base network generation
    model = getBaseModel(input_shape, nb_classes)

    model.summary()

    # %% model compilation and training
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)

    model.fit(x_train, y_train,
              epochs=50, callbacks=[EarlyStopping],
              validation_data=(x_test, y_test),
              verbose=1)
    