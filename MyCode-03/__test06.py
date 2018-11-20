import os
import time
import math
import numpy as np

import Metrics
import NpMetrics

from Database import INFO
from Database.Utils import reshape
from Database.Utils import load_data

from Schema.Utils import save_feature
from Schema.Utils import load_features

from Generator import MyDualV1

from scipy.stats import entropy
from scipy.spatial import distance

from tensorflow.keras import Model
from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":

    e_len = 128

    db = INFO['fashion']
    shape = db['shape']
    n_cls = db['n_cls']

    X_train, X_test, y_train, y_test = load_data('fashion')

    X_train = reshape(X_train / 255.0, shape)
    X_test = reshape(X_test / 255.0, shape)

    # %% CNN My Dual Model
    in_layer = layers.Input(shape=shape)

    conv_01 = layers.Conv2D(32, (3, 3))(in_layer)
    batch_01 = layers.BatchNormalization()(conv_01)
    active_01 = layers.Activation('relu')(batch_01)

    conv_02 = layers.Conv2D(32, (3, 3))(active_01)
    batch_02 = layers.BatchNormalization()(conv_02)
    active_02 = layers.Activation('relu')(batch_02)

    max2d_01 = layers.MaxPooling2D()(active_02)
    drop_01 = layers.Dropout(0.25)(max2d_01)
    flat_01 = layers.Flatten()(drop_01)

    dense_01 = layers.Dense(e_len, activation='sigmoid')(flat_01)
    drop_02 = layers.Dropout(0.5)(dense_01)
    out_layer = layers.Dense(n_cls, activation='softmax')(drop_02)

    inner_model = Model(inputs=in_layer, outputs=dense_01)
    outer_model = Model(inputs=in_layer, outputs=out_layer)

    in_layer_01 = layers.Input(shape=shape)
    in_layer_02 = layers.Input(shape=shape)

    embed_01 = inner_model(in_layer_01)
    embed_02 = inner_model(in_layer_02)

    out_01 = outer_model(in_layer_01)
    out_02 = outer_model(in_layer_02)

    concat = layers.Concatenate()([embed_01, out_01, embed_02, out_02])

    model = Model(inputs=[in_layer_01, in_layer_02], outputs=concat)

    # %% Trainable Model

    def dual_loss(n_cls):
        def _loss(y_true, y_pred):
            embed_01 = y_pred[:, :e_len]
            out_01 = y_pred[:, e_len:(e_len+n_cls)]
            tru_01 = y_true[:, :n_cls]

            embed_02 = y_pred[:, (e_len+n_cls):(e_len+n_cls+e_len)]
            out_02 = y_pred[:, (e_len+n_cls+e_len):(e_len+n_cls+e_len+n_cls)]
            tru_02 = y_true[:, n_cls:-1]

            true_emb = y_true[:, -1]

            embed_dist = Metrics.euclidean_distance(embed_01, embed_02)
            # embed_dist = Metrics.kullback_leibler(embed_01, embed_02) +\
            #     Metrics.kullback_leibler(embed_02, embed_01)
            # embed_dist = Metrics.cosine_similarity(embed_01, embed_02)

            pos_embed_dist = true_emb * embed_dist
            neg_embed_dist = (1-true_emb) * embed_dist

            loss = \
                Metrics.entropy(pos_embed_dist) +\
                Metrics.entropy(neg_embed_dist) +\
                Metrics.cross_entropy(tru_01, out_01) +\
                Metrics.cross_entropy(tru_02, out_02)

            return loss

        return _loss

    def my_accu(y_true, y_pred):
        out_01 = y_pred[:, e_len:(e_len+n_cls)]
        tru_01 = y_true[:, :n_cls]

        out_02 = y_pred[:, (e_len+n_cls+e_len):(e_len+n_cls+e_len+n_cls)]
        tru_02 = y_true[:, n_cls:-1]

        accu_01 = K.cast(
            K.equal(K.argmax(tru_01), K.argmax(out_01)), K.floatx())
        accu_02 = K.cast(
            K.equal(K.argmax(tru_02), K.argmax(out_02)), K.floatx())

        return (accu_01+accu_02)/2

    model.compile(loss=dual_loss(n_cls), optimizer='adadelta',
                  metrics=[my_accu])

    traingen = MyDualV1(X_train, y_train, n_cls)
    validgen = MyDualV1(X_test, y_test, n_cls)

    model.fit_generator(traingen, epochs=15, verbose=1, validation_data=validgen,
                        callbacks=[EarlyStopping(patience=50)],
                        workers=8, use_multiprocessing=True)

    def top_k_accuracy(y_score, y_true, k=5):
        argsrt = np.argsort(y_score)[:, -k:]
        top_k_bool = []
        for i in range(len(y_true)):
            if y_true[i] in argsrt[i]:
                top_k_bool.append(1)
            else:
                top_k_bool.append(0)
        return np.mean(top_k_bool)

    # %% Model for Analysis
    out_01_model = inner_model

    y_score = outer_model.predict(X_test)
    print('nn top 1 accu: %f' % (top_k_accuracy(y_score, y_test, 1)*100))
    print('nn top 3 accu: %f' % (top_k_accuracy(y_score, y_test, 3)*100))
    print('nn top 5 accu: %f' % (top_k_accuracy(y_score, y_test, 5)*100))

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

    _X, _y = _ext(X_train, y_train)
    embed_out_01 = out_01_model.predict(_X)

    X_01, y_01 = embed_out_01, _y

    # %% Embedding Analaysis

    print('\nmin')
    print(np.min(X_01))

    print('\nmax')
    print(np.max(X_01))

    print('\nmean')
    print(np.mean(X_01))

    print('\nstd')
    print(np.std(X_01), '\n\n')

    def _dist(func, X, y):
        dist = np.zeros((len(y), len(y)))
        for i in range(len(y)):
            for j in range(len(y)):
                dist[i, j] = func(X[i], X[j])
        return dist

    metrics = [
        (lambda a, b: distance.cosine(a, b), 'cosine'),
        (lambda a, b: distance.correlation(a, b), 'correlation'),
        (lambda a, b: distance.euclidean(a, b), 'euclidean'),
        (lambda a, b: distance.sqeuclidean(a, b), 'sqeuclidean'),
        (lambda a, b: NpMetrics.general_jaccard_similarity(
            a, b), 'general_jaccard_similarity'),
        (lambda a, b: NpMetrics.kullback_leibler(a, b), 'kullback_leibler'),
        (lambda a, b: NpMetrics.softmax_squared_l2_distance(
            a, b), 'softmax_squared_l2_distance'),
        (lambda a, b: NpMetrics.softmax_kullback_leibler(
            a, b), 'softmax_kullback_leibler'),
        (lambda a, b: NpMetrics.cross_entropy(a, b), 'cross_entropy'),
        (lambda a, b: NpMetrics.softmax_cross_entropy(
            a, b), 'softmax_cross_entropy'),
        (lambda a, b: NpMetrics.cross_entropy_loss(a, b), 'cross_entropy_loss'),
        (lambda a, b: NpMetrics.softmax_cross_entropy_loss(
            a, b), 'softmax_cross_entropy_loss'),
        (lambda a, b: NpMetrics.logistic_loss(a, b), 'logistic_loss'),
        (lambda a, b: NpMetrics.softmax_logistic_loss(
            a, b), 'softmax_logistic_loss'),
    ]

    def _analyze(X, y, title):
        for func, name in metrics:
            print(title+'_'+name)
            start = time.time()
            dist = _dist(func, X, y)
            print(time.time()-start, '\n')
            np.savetxt('./dists/'+title+'_'+name+'.csv', dist, delimiter=',')

    _analyze(X_01, y_01, '01')

    # %% kNN Analaysis
    train_embed = out_01_model.predict(X_train)
    test_embed = out_01_model.predict(X_test)

    clf = KNeighborsClassifier(
        weights='uniform', n_neighbors=1, n_jobs=8)
    clf.fit(train_embed, y_train)

    y_score = clf.predict_proba(test_embed)
    print('1 neighbor knn top 1 accu: %f' %
          (top_k_accuracy(y_score, y_test, 1)*100))

    clf = KNeighborsClassifier(
        weights='distance', n_neighbors=9, n_jobs=8)
    clf.fit(train_embed, y_train)

    y_score = clf.predict_proba(test_embed)
    print('knn top 1 accu: %f' % (top_k_accuracy(y_score, y_test, 1)*100))
    print('knn top 3 accu: %f' % (top_k_accuracy(y_score, y_test, 3)*100))
    print('knn top 5 accu: %f' % (top_k_accuracy(y_score, y_test, 5)*100))
