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

from Generator import MyTripletV1

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

    in_layer_anc = layers.Input(shape=shape)
    in_layer_pos = layers.Input(shape=shape)
    in_layer_neg = layers.Input(shape=shape)

    embed_anc = inner_model(in_layer_anc)
    embed_pos = inner_model(in_layer_pos)
    embed_neg = inner_model(in_layer_neg)

    out_anc = outer_model(in_layer_anc)
    out_pos = outer_model(in_layer_pos)
    out_neg = outer_model(in_layer_neg)

    concat = layers.Concatenate()([embed_anc, embed_pos, embed_neg,
                                   out_anc, out_pos, out_neg])

    model = Model(inputs=[in_layer_anc, in_layer_pos, in_layer_neg],
                  outputs=concat)

    # %% Trainable Model

    def dual_loss(n_cls):
        def _loss(y_true, y_pred):
            embed_anc = y_pred[:, :e_len]
            embed_pos = y_pred[:, e_len:(e_len*2)]
            embed_neg = y_pred[:, (e_len*2):(e_len*3)]

            out_anc = y_pred[:, (e_len*3):((e_len*3)+n_cls)]
            out_pos = y_pred[:, ((e_len*3)+n_cls):((e_len*3)+(n_cls*2))]
            out_neg = y_pred[:, ((e_len*3)+(n_cls*2)):((e_len*3)+(n_cls*3))]

            tru_anc = y_true[:, :n_cls]
            tru_pos = y_true[:, n_cls:(n_cls*2)]
            tru_neg = y_true[:, (n_cls*2):(n_cls*3)]

            # euclidean_distance --> embeding
            pos_embed_dist_eclid = Metrics.euclidean_distance(embed_anc, embed_pos)
            neg_embed_dist_eclid = Metrics.euclidean_distance(embed_anc, embed_neg)

            # euclidean_distance --> last layer
            pos_embed_dist_eclid_ll = Metrics.euclidean_distance(out_anc, out_pos)
            neg_embed_dist_eclid_ll = Metrics.euclidean_distance(out_anc, out_neg)

            # kullback_leibler --> embeding
            pos_embed_dist_kl = Metrics.kullback_leibler(embed_anc, embed_pos) +\
                Metrics.kullback_leibler(embed_pos, embed_anc)
            neg_embed_dist_kl = Metrics.kullback_leibler(embed_anc, embed_neg) +\
                Metrics.kullback_leibler(embed_neg, embed_anc)

            # kullback_leibler --> last layer
            pos_embed_dist_kl_ll = Metrics.kullback_leibler(out_anc, out_pos) +\
                Metrics.kullback_leibler(out_pos, out_anc)
            neg_embed_dist_kl_ll = Metrics.kullback_leibler(out_anc, out_neg) +\
                Metrics.kullback_leibler(out_neg, out_anc)

            # cosine_distance - cosine_similarity --> embeding
            pos_embed_dist_cosine = Metrics.cosine_distance(embed_anc, embed_pos)
            neg_embed_simi_cosine = Metrics.cosine_similarity(embed_anc, embed_neg)

            # cosine_distance - cosine_similarity --> last layer
            pos_embed_dist_cosine_ll = Metrics.cosine_distance(out_anc, out_pos)
            neg_embed_simi_cosine_ll = Metrics.cosine_similarity(out_anc, out_neg)

            zero = K.constant(0, dtype=K.floatx())
            one = K.constant(1, dtype=K.floatx())

            loss = \
                Metrics.entropy(K.tanh(pos_embed_dist_kl)) +\
                Metrics.entropy(K.tanh(neg_embed_dist_kl)) +\
                Metrics.entropy(K.tanh(pos_embed_dist_eclid)) +\
                Metrics.entropy(K.tanh(neg_embed_dist_eclid)) +\
                Metrics.cross_entropy(zero, pos_embed_dist_cosine) +\
                Metrics.cross_entropy(zero, neg_embed_simi_cosine) +\
                Metrics.cross_entropy(zero, K.tanh(pos_embed_dist_kl)) +\
                Metrics.cross_entropy(one, K.tanh(neg_embed_dist_kl)) +\
                Metrics.cross_entropy(zero, K.tanh(pos_embed_dist_eclid)) +\
                Metrics.cross_entropy(one, K.tanh(neg_embed_dist_eclid)) +\
                \
                pos_embed_dist_cosine_ll +\
                neg_embed_simi_cosine_ll +\
                Metrics.cross_entropy(tru_anc, out_anc) +\
                Metrics.cross_entropy(tru_pos, out_pos) +\
                Metrics.cross_entropy(tru_neg, out_neg)
                # pos_embed_dist_kl_ll +\
                # pos_embed_dist_eclid_ll +\
                # \
                # Metrics.entropy(K.tanh(pos_embed_dist_kl_ll)) +\
                # Metrics.entropy(K.tanh(neg_embed_dist_kl_ll)) +\
                # Metrics.entropy(K.tanh(pos_embed_dist_eclid_ll)) +\
                # Metrics.entropy(K.tanh(neg_embed_dist_eclid_ll)) +\
                # Metrics.cross_entropy(zero, pos_embed_dist_cosine_ll) +\
                # Metrics.cross_entropy(zero, neg_embed_simi_cosine_ll) +\
                # Metrics.cross_entropy(zero, K.tanh(pos_embed_dist_kl_ll)) +\
                # Metrics.cross_entropy(one, K.tanh(neg_embed_dist_kl_ll)) +\
                # Metrics.cross_entropy(zero, K.tanh(pos_embed_dist_eclid_ll)) +\
                # Metrics.cross_entropy(one, K.tanh(neg_embed_dist_eclid_ll)) +\

            return loss

        return _loss

    def my_accu(y_true, y_pred):
        out_anc = y_pred[:, (e_len*3):((e_len*3)+n_cls)]
        out_pos = y_pred[:, ((e_len*3)+n_cls):((e_len*3)+(n_cls*2))]
        out_neg = y_pred[:, ((e_len*3)+(n_cls*2)):((e_len*3)+(n_cls*3))]

        tru_anc = y_true[:, :n_cls]
        tru_pos = y_true[:, n_cls:(n_cls*2)]
        tru_neg = y_true[:, (n_cls*2):(n_cls*3)]

        accu_anc = K.cast(
            K.equal(K.argmax(tru_anc), K.argmax(out_anc)), K.floatx())
        accu_pos = K.cast(
            K.equal(K.argmax(tru_pos), K.argmax(out_pos)), K.floatx())
        accu_neg = K.cast(
            K.equal(K.argmax(tru_neg), K.argmax(out_neg)), K.floatx())

        return (accu_anc+accu_pos+accu_neg)/3

    model.compile(loss=dual_loss(n_cls), optimizer='adadelta',
                  metrics=[my_accu])

    traingen = MyTripletV1(X_train, y_train, n_cls)
    validgen = MyTripletV1(X_test, y_test, n_cls)

    model.fit_generator(traingen, epochs=20, verbose=1, validation_data=validgen,
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
    embed_out_02 = outer_model.predict(_X)

    X_01, y_01 = embed_out_01, _y
    X_02, y_02 = embed_out_02, _y

    # %% Embedding Analaysis

    print('\nmin')
    print(np.min(X_01))
    print(np.min(X_02))

    print('\nmax')
    print(np.max(X_01))
    print(np.max(X_02))

    print('\nmean')
    print(np.mean(X_01))
    print(np.mean(X_02))

    print('\nstd')
    print(np.std(X_01))
    print(np.std(X_02), '\n\n')

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
    _analyze(X_02, y_02, '02')

    # %% kNN Analaysis
    def _kNN_analyze(out_model):
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

    print('out_01_model')
    _kNN_analyze(out_01_model)
    print('\nouter_model')
    _kNN_analyze(outer_model)