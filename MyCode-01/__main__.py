# pylint: disable=W0614
import Utils
import Loader

import Estimator

from Config import Conf
from Config import Estm

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from Losses import triplet
from Losses import contrastive
from Losses import cross_entropy

import numpy as np

from Generator import General


# %% Main program


def NotAugmentedRun(db, version, estm, loss, optimizer, metric=[], callback=[], verbose=1):
    schema = Loader.getSchema(db, version, estm)
    schema.summary()
    schema.plot(estm)

    estimator = getattr(Estimator, estm)
    estimator = estimator(schema.getModel())
    estimator.compile(loss=loss,
                      optimizer=optimizer,
                      metric=metric)
    history = estimator.fit(db, verbose=verbose,
                            callbacks=callback)

    history.plot(estimator.name+'_'+db.name+'_NotAugmented')

    print(estimator.evaluate(db.X_train, db.Y_train()))
    y_pred = np.argmax(estimator.predict(db.X_train), axis=-1)
    print(Utils.classificationReport('train', db.y_train, y_pred))

    print(estimator.evaluate(db.X_test, db.Y_test()))
    y_pred = np.argmax(estimator.predict(db.X_test), axis=-1)
    print(Utils.classificationReport('test', db.y_test, y_pred))

    Utils.rocCurve(db.name+'_'+schema.name+'_'+estimator.name+'_NotAugmented',
                   db.Y_test(), estimator.predict(db.X_test), db.info['n_cls'])

    schema.saveWeights(estimator.name+'_'+db.name+'_NotAugmented')
    schema.extract(estimator.name+'_NotAugmented', db)


def AugmentedRun(db, generator, version, estm, loss, optimizer, metric=[], callback=[], verbose=1):
    schema = Loader.getSchema(db, version, estm)
    schema.summary()
    schema.plot(estm)

    estimator = getattr(Estimator, estm)
    estimator = estimator(schema.getModel())
    estimator.compile(loss=loss,
                      optimizer=optimizer,
                      metric=metric)
    history = estimator.fit_on_batch(db, generator, verbose=verbose,
                                     callbacks=callback)

    history.plot(estimator.name+'_'+db.name+'_Augmented')

    print(estimator.evaluate(db.X_train, db.Y_train()))
    y_pred = np.argmax(estimator.predict(db.X_train), axis=-1)
    print(Utils.classificationReport('train', db.y_train, y_pred))

    print(estimator.evaluate(db.X_test, db.Y_test()))
    y_pred = np.argmax(estimator.predict(db.X_test), axis=-1)
    print(Utils.classificationReport('test', db.y_test, y_pred))

    Utils.rocCurve(db.name+'_'+schema.name+'_'+estimator.name+'_Augmented',
                   db.Y_test(), estimator.predict(db.X_test), db.info['n_cls'])

    schema.saveWeights(estimator.name+'_'+db.name+'_Augmented')
    schema.extract(estimator.name+'_Augmented', db)


for data, version, augment in Conf:
    db = Loader.getDataset(data)
    db = Utils.preProcessing(db)
    # db.histogram()
    db.summary()

    # %% Conventional
    # NotAugmentedRun(db, version, Estm.Conventional,
    #                 losses.categorical_crossentropy,
    #                 optimizers.Adadelta(), ['acc'],
    #                 [callbacks.EarlyStopping(patience=20)])

    generator = General(X_train=db.X_train, y_train=db.y_train,
                        augment=True, allowable=augment)
    AugmentedRun(db, generator,
                 version, Estm.Conventional,
                 losses.categorical_crossentropy,
                 optimizers.Adadelta(), ['acc'],
                 [callbacks.EarlyStopping(patience=20)])
