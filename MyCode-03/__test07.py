import os
import ray
import time
import math
import numpy as np

import NpMetrics as Metrics

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

from sklearn.svm import SVC
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":

    ray.init()

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

    conv_out_01 = active_01

    conv_02 = layers.Conv2D(32, (3, 3))(active_01)
    batch_02 = layers.BatchNormalization()(conv_02)
    active_02 = layers.Activation('relu')(batch_02)

    max2d_01 = layers.MaxPooling2D()(active_02)

    conv_out_02 = max2d_01

    drop_01 = layers.Dropout(0.25)(max2d_01)
    flat_01 = layers.Flatten()(drop_01)

    dense_01 = layers.Dense(128, activation='sigmoid')(flat_01)

    conv_out_03 = layers.Flatten()(dense_01)

    drop_02 = layers.Dropout(0.5)(dense_01)
    out_layer = layers.Dense(n_cls, activation='softmax')(drop_02)

    # %% Trainable Model
    org_model = Model(inputs=in_layer, outputs=out_layer)

    org_model.summary()

    org_model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta', metrics=['acc'])

    org_model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=1000, callbacks=[
        EarlyStopping(patience=50)], validation_data=(X_test, to_categorical(y_test)), verbose=2)

    def top_k_accuracy(y_score, y_true, k=5):
        argsrt = np.argsort(y_score)[:, -k:]
        top_k_bool = []
        for i in range(len(y_true)):
            if y_true[i] in argsrt[i]:
                top_k_bool.append(1)
            else:
                top_k_bool.append(0)
        return np.mean(top_k_bool)

    y_score = org_model.predict(X_test)
    print('nn top 1 accu: %f' % (top_k_accuracy(y_score, y_test, 1)*100))
    print('nn top 3 accu: %f' % (top_k_accuracy(y_score, y_test, 3)*100))
    print('nn top 5 accu: %f' % (top_k_accuracy(y_score, y_test, 5)*100))

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

    _X, _y = _ext(X_train, y_train)
    embed_out_01 = out_01_model.predict(_X)
    embed_out_02 = out_02_model.predict(_X)
    embed_out_03 = out_03_model.predict(_X)
    embed_out_04 = org_model.predict(_X)

    X_01, y_01 = embed_out_01, _y
    X_02, y_02 = embed_out_02, _y
    X_03, y_03 = embed_out_03, _y
    X_04, y_04 = embed_out_04, _y

    # %% Embedding Analaysis

    print('\nmin')
    print(np.min(X_01))
    print(np.min(X_02))
    print(np.min(X_03))
    print(np.min(X_04))

    print('\nmax')
    print(np.max(X_01))
    print(np.max(X_02))
    print(np.max(X_03))
    print(np.max(X_04))

    print('\nmean')
    print(np.mean(X_01))
    print(np.mean(X_02))
    print(np.mean(X_03))
    print(np.mean(X_04))

    print('\nstd')
    print(np.std(X_01))
    print(np.std(X_02))
    print(np.std(X_03))
    print(np.std(X_04), '\n\n')

    def _cnn_dist(func, X, y):
        @ray.remote
        def function(a, b):
            avg = []
            for i in range(a.shape[-1]):
                avg.append(func(a[:, :, i],
                                b[:, :, i]))
            return np.mean(avg)
        dist = np.zeros((len(y), len(y))).tolist()
        for i in range(len(y)):
            for j in range(len(y)):
                dist[i][j] = function.remote(X[i], X[j])
        for i in range(len(y)):
            dist[i] = ray.get(dist[i])
        return np.array(dist)

    def _dist(func, X, y):
        @ray.remote
        def function(a, b):
            return func(a, b)
        dist = np.zeros((len(y), len(y))).tolist()
        for i in range(len(y)):
            for j in range(len(y)):
                dist[i][j] = function.remote(X[i], X[j])
        for i in range(len(y)):
            dist[i] = ray.get(dist[i])
        return np.array(dist)

    metrics = [
        (lambda a, b: distance.cosine(a, b), 'cosine'),
        (lambda a, b: distance.correlation(a, b), 'correlation'),
        (lambda a, b: distance.euclidean(a, b), 'euclidean'),
        (lambda a, b: distance.sqeuclidean(a, b), 'sqeuclidean'),
        (lambda a, b: Metrics.general_jaccard_similarity(
            a, b), 'general_jaccard_similarity'),
        (lambda a, b: Metrics.kullback_leibler(a, b), 'kullback_leibler'),
        (lambda a, b: Metrics.softmax_squared_l2_distance(
            a, b), 'softmax_squared_l2_distance'),
        (lambda a, b: Metrics.softmax_kullback_leibler(
            a, b), 'softmax_kullback_leibler'),
        (lambda a, b: Metrics.cross_entropy(a, b), 'cross_entropy'),
        (lambda a, b: Metrics.softmax_cross_entropy(a, b), 'softmax_cross_entropy'),
        (lambda a, b: Metrics.cross_entropy_loss(a, b), 'cross_entropy_loss'),
        (lambda a, b: Metrics.softmax_cross_entropy_loss(
            a, b), 'softmax_cross_entropy_loss'),
        (lambda a, b: Metrics.logistic_loss(a, b), 'logistic_loss'),
        (lambda a, b: Metrics.softmax_logistic_loss(a, b), 'softmax_logistic_loss'),
        (lambda a, b: Metrics.softmax_logistic_loss(a, b), 'softmax_logistic_loss'),
    ]

    cnn_metrics = [
        # (lambda a, b: Metrics.mutual_information(a, b, 256), 'mutual_information'),
        # (lambda a, b: np.sum(np.matrix(a)*np.matrix(b)), 'my_metric_v01'),
        (lambda a, b: np.sum(((np.matrix(a)*np.matrix(b))-(np.matrix(a)*np.matrix(a))) +
                             ((np.matrix(b)*np.matrix(a))-(np.matrix(b)*np.matrix(b)))), 'my_metric_v02'),
    ]

    def _analyze(X, y, title):
        for func, name in metrics:
            print(title+'_'+name)
            start = time.time()
            dist = _dist(func, X, y)
            print(time.time()-start, '\n')
            np.savetxt('./dists/'+title+'_'+name+'.csv', dist, delimiter=',')

    def _cnn_analyze(X, y, title):
        # X = np.tanh(X)
        for func, name in cnn_metrics:
            print(title+'_'+name)
            start = time.time()
            dist = _cnn_dist(func, X, y)
            print(time.time()-start, '\n')
            np.savetxt('./dists/'+title+'_'+name+'.csv', dist, delimiter=',')

    # _cnn_analyze(X_01, y_01, '01')
    # _cnn_analyze(X_02, y_02, '02')
    # _analyze(X_01, y_01, '01')
    # _analyze(X_02, y_02, '02')
    _analyze(X_03, y_03, '03')
    _analyze(X_04, y_04, '04')

    # %% kNN Analaysis

    def _kNN_analyze_v1(out_model):
        train_embed = out_model.predict(X_train)
        test_embed = out_model.predict(X_test)

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

    def _kNN_analyze_v2(out_model):
        train_embed = out_model.predict(X_train)
        train_embed = train_embed/np.sqrt(np.sum(np.square(train_embed), axis=-1, keepdims=True))
        test_embed = out_model.predict(X_test)
        test_embed = test_embed/np.sqrt(np.sum(np.square(test_embed), axis=-1, keepdims=True))

        def myMetric(a, b):
            return 1.0-np.sum(a * b)

        clf = KNeighborsClassifier(metric='pyfunc', metric_params={'func': myMetric},
                                   weights='uniform', n_neighbors=1, n_jobs=8)
        clf.fit(train_embed, y_train)

        y_score = clf.predict_proba(test_embed)
        print('1 neighbor knn top 1 accu: %f' %
              (top_k_accuracy(y_score, y_test, 1)*100))

        clf = KNeighborsClassifier(metric='pyfunc', metric_params={'func': myMetric},
                                   weights='distance', n_neighbors=9, n_jobs=8)
        clf.fit(train_embed, y_train)

        y_score = clf.predict_proba(test_embed)
        print('knn top 1 accu: %f' % (top_k_accuracy(y_score, y_test, 1)*100))
        print('knn top 3 accu: %f' % (top_k_accuracy(y_score, y_test, 3)*100))
        print('knn top 5 accu: %f' % (top_k_accuracy(y_score, y_test, 5)*100))

    print('out_03_model')
    _kNN_analyze_v1(out_03_model)
    print('\norg_model')
    _kNN_analyze_v1(org_model)
    print('\nv2 cosine distance\nout_03_model')
    _kNN_analyze_v2(out_03_model)
    print('\norg_model')
    _kNN_analyze_v2(org_model)

    def _svm_v1(out_model):
        train_embed = out_model.predict(X_train)
        test_embed = out_model.predict(X_test)

        def _report(clf):
            clf.fit(train_embed, y_train)
            y_score = clf.predict_proba(test_embed)
            print('top 1 accu: %f' % (top_k_accuracy(y_score, y_test, 1)*100))
            print('top 3 accu: %f' % (top_k_accuracy(y_score, y_test, 3)*100))
            print('top 5 accu: %f' % (top_k_accuracy(y_score, y_test, 5)*100))

        print('linear kernel')
        clf = SVC(kernel='linear', gamma='scale', probability=True)  # Linear Kernel
        _report(clf)
        print('rbf kernel')
        clf = SVC(kernel='rbf', gamma='scale', probability=True)  # Linear Kernel
        _report(clf)
        print('poly 3 kernel')
        clf = SVC(kernel='poly', gamma='scale', probability=True)  # Linear Kernel
        _report(clf)
        print('sigmoid kernel')
        clf = SVC(kernel='sigmoid', gamma='scale', probability=True)  # Linear Kernel
        _report(clf)

    print('out_03_model')
    _svm_v1(out_03_model)
    print('\norg_model')
    _svm_v1(org_model)
