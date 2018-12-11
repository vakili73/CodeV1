import numpy as np
from Reporter import Report

from Config import FITOPTS
from Config import PATIENCE
from Config import OPTIMIZER
from Config import BATCHSIZE
from Config import FITGENOPTS

from Utils import load_loss
from Utils import load_datagen
from Utils import load_metrics

from Schema import load_schema

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


def Run(rpt: Report, bld: str, n_cls: int, shape: tuple, db_opt: dict, bld_opt: dict,
        X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray, aug_flag: bool = False):

    schm = db_opt['schema']
    schema = load_schema(schm)
    dgen_opt = db_opt['dgen_opt']
    getattr(schema, 'build'+bld)(shape, n_cls)
    schema.model.summary()

    is_clf = True if 'classification' in bld_opt else False
    datagen = load_datagen(bld_opt['datagen'])

    metric_args = getMetricArgs(bld, n_cls, schema)
    metrics = load_metrics(bld_opt['metrics'], **metric_args)

    loss_args = getLossArgs(bld, n_cls, schema)
    loss = load_loss(bld_opt['loss'], **loss_args)
    knn_opt = bld_opt['knn']
    svm_opt = bld_opt['svm']

    rpt.write_build(bld).write_schema(schm).flush()
    rpt.write_augment(aug_flag).flush()

    schema.model.compile(loss=loss, optimizer=OPTIMIZER, metrics=metrics)

    history = fitModel(schema, n_cls, dgen_opt, datagen,
                       X_train, X_valid, y_train, y_valid, aug_flag)

    rpt.end_line()


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
    return history.history
