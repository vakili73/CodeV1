import os
import time
import math
import numpy as np

from Database import INFO
from Database.Utils import reshape
from Database.Utils import load_data

from Schema.Utils import save_feature
from Schema.Utils import load_features

from scipy.stats import entropy
from scipy.spatial import distance

from tensorflow.keras import Model
from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


db = INFO['fashion']
shape = db['shape']
n_cls = db['n_cls']

X_train, X_test, y_train, y_test = load_data('fashion')
X_train = reshape(X_train / 255.0, shape)
X_test = reshape(X_test / 255.0, shape)

# %% CNN Model
in_layer = layers.Input(shape=shape)

conv_01 = layers.Conv2D(32, (3, 3))(in_layer)
batch_01 = layers.BatchNormalization()(conv_01)
active_01 = layers.Activation('relu')(batch_01)

conv_out_01 = layers.Flatten()(active_01)

conv_02 = layers.Conv2D(32, (3, 3))(active_01)
batch_02 = layers.BatchNormalization()(conv_02)
active_02 = layers.Activation('relu')(batch_02)

conv_out_02 = layers.Flatten()(active_02)

max2d_01 = layers.MaxPooling2D()(active_02)
drop_01 = layers.Dropout(0.25)(max2d_01)
flat_01 = layers.Flatten()(drop_01)

dense_01 = layers.Dense(128, activation='relu')(flat_01)

conv_out_03 = layers.Flatten()(dense_01)

drop_02 = layers.Dropout(0.5)(dense_01)
out_layer = layers.Dense(n_cls, activation='softmax')(drop_02)

# %% Trainable Model
org_model = Model(inputs=in_layer, outputs=out_layer)

org_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta', metrics=['acc'])

if not os.path.exists('./logs/features'):
    org_model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=1000, callbacks=[
        EarlyStopping(patience=50)], validation_data=(X_test, to_categorical(y_test)))

# %% Model for Analysis
out_01_model = Model(inputs=in_layer, outputs=conv_out_01)
out_02_model = Model(inputs=in_layer, outputs=conv_out_02)
out_03_model = Model(inputs=in_layer, outputs=conv_out_03)

# %% Embeddings for training data


def _ext(X_data, y_data):
    sorted_X = []
    sorted_y = []
    for i in range(n_cls):
        idx = np.where(y_data == i)[0][:5]
        sorted_X.append(X_data[idx])
        sorted_y.append(y_data[idx])
    return np.concatenate(
        sorted_X), np.concatenate(sorted_y)


if not os.path.exists('./logs/features'):
    _X, _y = _ext(X_train, y_train)
    embed_out_01 = out_01_model.predict(_X)
    embed_out_02 = out_02_model.predict(_X)
    embed_out_03 = out_03_model.predict(_X)

    save_feature(embed_out_01, _y, 'embed_out_01')
    save_feature(embed_out_02, _y, 'embed_out_02')
    save_feature(embed_out_03, _y, 'embed_out_03')
else:
    embed_out_01, _y = load_features('embed_out_01.txt')
    embed_out_02, _y = load_features('embed_out_02.txt')
    embed_out_03, _y = load_features('embed_out_03.txt')

X_01, y_01 = embed_out_01, _y
X_02, y_02 = embed_out_02, _y
X_03, y_03 = embed_out_03, _y

# %% Embedding Analaysis

print(np.min(X_01))
print(np.min(X_02))
print(np.min(X_03))

print(np.max(X_01))
print(np.max(X_02))
print(np.max(X_03))

print(np.mean(X_01))
print(np.mean(X_02))
print(np.mean(X_03))

print(np.std(X_01))
print(np.std(X_02))
print(np.std(X_03))


def _dist(func, X, y):
    dist = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            dist[i, j] = func(X[i], X[j])
    return dist


def general_jaccard(a, b):
    _a = np.sum(np.min(np.stack((a, b)), axis=0))
    _b = np.sum(np.max(np.stack((a, b)), axis=0))
    return _a/_b


metrics = [
    (lambda a, b: math.exp(-math.log(np.maximum(1.0 - distance.cosine(a, b), 1e-7))), 'exp_log_cosine_dist'),
    (lambda a, b: -math.log(np.maximum(1.0 - distance.correlation(a, b), 1e-7)), 'log_correlation_dist'),
    (lambda a, b: math.exp(math.log(np.maximum(distance.euclidean(a, b), 1e-7))), 'exp_log_euclidean_dist'),
    (lambda a, b: math.exp(math.log10(np.maximum(distance.sqeuclidean(a, b), 1e-7))), 'exp_log_sqeuclidean_dist'),
    (lambda a, b: math.exp(-math.log(np.maximum(general_jaccard(a, b), 1e-7))), 'exp_log_general-jaccard_dist'),
    # (lambda a, b: entropy(a, b), 'kullback-leibler_dist'),
]


def _analyze(X, y, title):
    for func, name in metrics:
        print(title+'_'+name)
        start = time.time()
        dist = _dist(func, X, y)
        print(time.time()-start)
        np.savetxt('./dists/'+title+'_'+name+'.csv', dist, delimiter=',')


_analyze(X_01, y_01, '01')
_analyze(X_02, y_02, '02')
_analyze(X_03, y_03, '03')
