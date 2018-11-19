import os
import time
import math
import numpy as np

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

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":

    embed_size = 128

    db = INFO['fashion']
    shape = db['shape']
    n_cls = db['n_cls']

    X_train, X_test, y_train, y_test = load_data('fashion')

    y_tsne = TSNE(n_components=3).fit_transform(X_train / 255.0)

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

    dense_01 = layers.Dense(embed_size, activation='sigmoid')(flat_01)
    drop_02 = layers.Dropout(0.5)(dense_01)
    out_layer = layers.Dense(3, activation='sigmoid')(drop_02)

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
            embed_01 = y_pred[:, :embed_size]
            out_01 = y_pred[:, embed_size:(embed_size+n_cls)]
            tru_01 = y_true[:, :n_cls]

            embed_02 = y_pred[:, (embed_size+n_cls):(embed_size+n_cls+embed_size)]
            out_02 = y_pred[:, (embed_size+n_cls+embed_size):(embed_size+n_cls+embed_size+n_cls)]
            tru_02 = y_true[:, n_cls:-1]

            true_emb = y_true[:, -1]

            def kullback_leibler_divergence(a, b):
                _a = K.clip(a, K.epsilon(), 1)
                _b = K.clip(b, K.epsilon(), 1)

                divergence = true_emb * (K.sum(K.square(_a - _b), axis=-1) +
                                         K.sum(_a * K.log(_a / _b), axis=-1))
                # (1-true_emb) * -K.sum(K.square(_a - _b), axis=-1)
                # (K.l2_normalize(_a, axis=-1)*K.l2_normalize(_b, axis=-1))
                # (K.sum(_a * K.log(_a / _b), axis=-1) +
                #                          K.sum(_b * K.log(_b / _a), axis=-1) +
                #                          K.sum(K.square(_a - _b), axis=-1)) +\
                return divergence

            tru_01 = K.clip(tru_01, K.epsilon(), 1)
            out_01 = K.clip(out_01, K.epsilon(), 1)
            loss = K.sum(tru_01 * K.log(tru_01 / out_01), axis=-1) +\
                K.sum(out_01 * K.log(out_01 / tru_01), axis=-1) +\
                K.sum(K.square(tru_01 - out_01), axis=-1)
            # kullback_leibler_divergence(embed_01, embed_02) +\
            #     K.categorical_crossentropy(tru_01, out_01, from_logits=True) +\
            #     K.categorical_crossentropy(tru_02, out_02, from_logits=True)
            # K.sum(tru_01 * K.log(tru_01 / out_01), axis=-1)+K.sum(K.square(tru_01 - out_01), axis=-1) +\

            return loss

        return _loss

    def my_accu(y_true, y_pred):
        out_01 = y_pred[:, embed_size:(embed_size+n_cls)]
        tru_01 = y_true[:, :n_cls]

        out_02 = y_pred[:, (embed_size+n_cls+embed_size):(embed_size+n_cls+embed_size+n_cls)]
        tru_02 = y_true[:, n_cls:-1]

        accu_01 = K.cast(
            K.equal(K.argmax(tru_01), K.argmax(out_01)), K.floatx())
        accu_02 = K.cast(
            K.equal(K.argmax(tru_02), K.argmax(out_02)), K.floatx())

        return (accu_01+accu_02)/2

    model.compile(loss=dual_loss(n_cls), optimizer='adadelta',
                  metrics=['mse'])

    traingen = MyDualV1(X_train, y_tsne, n_cls)

    model.fit_generator(traingen, epochs=10, verbose=1,
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

    def general_jaccard(a, b):
        _a = np.sum(np.min(np.stack((a, b)), axis=0))
        _b = np.sum(np.max(np.stack((a, b)), axis=0))
        return _a/_b

    def kullback_leibler(a, b):
        a = np.clip(abs(a), 1e-7, 1.0)
        b = np.clip(abs(b), 1e-7, 1.0)
        return entropy(a, b)

    metrics = [
        (lambda a, b: distance.cosine(a, b), 'cosine_dist'),
        (lambda a, b: distance.correlation(a, b), 'correlation_dist'),
        (lambda a, b: distance.euclidean(a, b), 'euclidean_dist'),
        (lambda a, b: distance.sqeuclidean(a, b), 'sqeuclidean_dist'),
        (lambda a, b: general_jaccard(a, b), 'general-jaccard_dist'),
        (lambda a, b: kullback_leibler(a, b), 'kullback-leibler_dist'),
        (lambda a, b: math.exp(-math.log(np.maximum(1.0 -
                                                    distance.cosine(a, b), 1e-7))), 'exp_log_cosine_dist'),
        (lambda a, b: -math.log(np.maximum(1.0 - distance.correlation(a, b), 1e-7)),
         'log_correlation_dist'),
        (lambda a, b: math.exp(math.log(np.maximum(
            distance.euclidean(a, b), 1e-7))), 'exp_log_euclidean_dist'),
        (lambda a, b: math.exp(math.log10(np.maximum(
            distance.sqeuclidean(a, b), 1e-7))), 'exp_log_sqeuclidean_dist'),
        (lambda a, b: math.exp(-math.log(np.maximum(general_jaccard(a, b), 1e-7))),
         'exp_log_general-jaccard_dist'),
        (lambda a, b: math.exp(-math.log(np.maximum(1.0 -
                                                    kullback_leibler(a, b), 1e-7))), 'exp_log_kullback-leibler_dist'),
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
