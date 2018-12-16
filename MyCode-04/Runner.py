import numpy as np
from Reporter import Report

from Config import FITOPTS
from Config import PATIENCE
from Config import OPTIMIZER
from Config import BATCHSIZE
from Config import FITGENOPTS
from Config import TOP_K_ACCU

from Utils import load_loss
from Utils import load_datagen
from Utils import load_metrics
from Utils import save_history
from Utils import plot_lr_curve
from Utils import plot_reduction
from Utils import plot_roc_curve
from Utils import plot_confusion_matrix

from Schema import load_schema
from Schema.Utils import save_weights
from Schema.Utils import save_feature

from Metrics import top_k_accuracy

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from scipy.spatial.distance import cosine

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


def Run(rpt: Report, bld: str, n_cls: int, shape: tuple, db_opt: dict, bld_opt: dict,
        X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray, db: str, shot: int, aug_flag: bool = False):

    schm = db_opt['schema']
    schema = load_schema(schm)
    dgen_opt = db_opt['dgen_opt']
    getattr(schema, 'build'+bld)(shape, n_cls)
    schema.model.summary()

    is_clf = True if 'classification' in bld_opt else False
    if aug_flag:
        datagen = load_datagen('Aug'+bld_opt['datagen'])
    else:
        datagen = load_datagen(bld_opt['datagen'])

    metric_args = getMetricArgs(bld, n_cls, schema)
    metrics = load_metrics(bld_opt['metrics'], **metric_args)

    loss_args = getLossArgs(bld, n_cls, schema)
    loss = load_loss(bld_opt['loss'], **loss_args)
    knn_opt = bld_opt['knn']
    svm_opt = bld_opt['svm']

    title = db+'_'+bld+'_'+schm+'_'+str(aug_flag)+'_'+str(shot)
    print(title)
    rpt.write_build(bld).write_schema(schm).flush()
    rpt.write_augment(aug_flag).flush()

    schema.model.compile(loss=loss, optimizer=OPTIMIZER, metrics=metrics)

    history = fitModel(schema, n_cls, dgen_opt, datagen,
                       X_train, X_valid, y_train, y_valid, aug_flag)
    save_history(history, title)
    save_weights(schema.model, title)
    plot_lr_curve(history, title)

    embed_feature = getFeatures(schema.getModel(), X_test)
    save_feature(embed_feature, y_test, title)
    plot_reduction(embeddings=embed_feature, targets=y_test,
                   title=title)

    if is_clf:
        rpt.write_text('nn_metrics').flush()
        if isinstance(embed_feature, list):
            y_score = embed_feature[-1]
            clf_report(rpt, y_test, y_score, n_cls, title)
        else:
            y_score = schema.model.predict(X_test)
            clf_report(rpt, y_test, y_score, n_cls, title)

    embed_feature_train = getFeatures(schema.getModel(), X_train)
    if isinstance(embed_feature, list):
        knn_opt = getKnnOpts(bld, knn_opt)
        for i in range(len(knn_opt)):
            rpt.write_text('knn_metrics_%d' % i).flush()
            Run_KNN(rpt, knn_opt[i], embed_feature_train[i],
                    embed_feature[i], y_train, y_test, n_cls, title+'_'+str(i))
    else:
        rpt.write_text('knn_metrics').flush()
        Run_KNN(rpt, knn_opt, embed_feature_train,
                embed_feature, y_train, y_test, n_cls, title)

    if isinstance(embed_feature, list):
        for i in range(len(embed_feature)):
            rpt.write_text('svm_metrics_%d' % i).flush()
            Run_SVM(rpt, svm_opt, embed_feature_train[i],
                    embed_feature[i], y_train, y_test, n_cls, title+'_'+str(i))
    else:
        rpt.write_text('svm_metrics').flush()
        Run_SVM(rpt, svm_opt, embed_feature_train,
                embed_feature, y_train, y_test, n_cls, title)

    rpt.end_line()


def Run_SVM(rpt, svm_opt, X_train,
            X_test, y_train, y_test, n_cls, title):
    for svm in svm_opt:
        _title = title + \
            '_kernel_{}'.format(svm['kernel'])
        print(_title)
        rpt.write_svm(svm['kernel']).flush()
        clf = SVC(**svm, probability=True)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        clf_report(rpt, y_test, y_score, n_cls, _title)


def Run_KNN(rpt, knn_opt, X_train,
            X_test, y_train, y_test, n_cls, title):
    for knn in knn_opt:
        _title = title + \
            '_weights_{}_neighbors_{}'.format(
                knn['weights'], knn['n_neighbors'])
        print(_title)
        rpt.write_knn(knn['weights'], knn['n_neighbors']).flush()
        clf = KNeighborsClassifier(**knn)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        clf_report(rpt, y_test, y_score, n_cls, _title)


def getKnnOpts(bld: str, knn_opt: dict):
    _knn_opt = []
    if bld.startswith('MyModel'):
        _knn_opt.append(knn_opt['embed_layer'])
        _knn_opt.append([])
        for knn in knn_opt['output_layer'][0]:
            _knn = dict(knn)
            _knn.update(knn_opt['output_layer'][1])
            _knn_opt[1].append(_knn)
    return _knn_opt


def getMetricArgs(bld: str, n_cls, schema):
    m_args = {'n_cls': n_cls}
    if bld.startswith('MyModel'):
        m_args.update({'e_len': schema.e_len})
    return m_args


def getLossArgs(bld: str, n_cls, schema):
    l_args = {'n_cls': n_cls}
    if bld.startswith('MyModel'):
        l_args.update({'e_len': schema.e_len})
    return l_args


def clf_report(rpt: Report, y_true, y_score, n_cls, title):
    y_pred = np.argmax(y_score, axis=-1)
    print(classification_report(y_true, y_pred, digits=5))
    plot_roc_curve(title, to_categorical(y_true, n_cls), y_score, n_cls)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, title, np.unique(y_true))
    for k in TOP_K_ACCU:
        score = top_k_accuracy(y_score, y_true, k)
        rpt.write_top_k_accuracy(score, k).flush()
    result = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    rpt.write_precision_recall_f1score(result[0], result[1], result[2])


def getFeatures(model, X):
    if isinstance(model, list):
        features = []
        for mdl in model:
            features.append(mdl.predict(X))
    else:
        features = model.predict(X)
    return features


def fitModel(schema, n_cls, dgen_opt, datagen,
             X_train, X_valid, y_train, y_valid, aug_flag):
    callbacks = [EarlyStopping(patience=PATIENCE),
                 TerminateOnNaN()]
    if datagen == 'Original':
        if aug_flag:
            datagen = ImageDataGenerator(**dgen_opt)
            datagen.fit(X_train)
            traingen = datagen.flow(X_train, to_categorical(y_train, n_cls),
                                    batch_size=BATCHSIZE)
            validgen = datagen.flow(X_valid, to_categorical(y_valid, n_cls),
                                    batch_size=BATCHSIZE)
        else:
            history = schema.model.fit(
                X_train, to_categorical(y_train, n_cls), **FITOPTS, callbacks=callbacks,
                validation_data=(X_valid, to_categorical(y_valid, n_cls)))
            history.history.update({'params': history.params})
            history.history.update({'epoch': history.epoch})
            return history.history
    else:
        if aug_flag:
            traingen = datagen(X_train, y_train, n_cls, dgen_opt, BATCHSIZE)
            validgen = datagen(X_valid, y_valid, n_cls, dgen_opt, BATCHSIZE)
        else:
            traingen = datagen(X_train, y_train, n_cls, BATCHSIZE)
            validgen = datagen(X_valid, y_valid, n_cls, BATCHSIZE)

    history = schema.model.fit_generator(traingen, validation_data=validgen,
                                         callbacks=callbacks, **FITGENOPTS)
    history.history.update({'params': history.params})
    history.history.update({'epoch': history.epoch})
    return history.history
