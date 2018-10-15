# pylint: disable=W0614
import Utils
import Loader

from Estimator import *

from Config import Conf
from Config import Estm

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from Losses import triplet_loss
from Losses import contrastive_loss

import numpy as np


# %% Main program

for data, version in Conf:
    db = Loader.getDataset(data)
    db = Utils.preProcessing(db)
    db.histogram()
    db.summary()

    # %% ConvenientNN
    schema = Loader.getSchema(db, version, Estm.CNN)
    schema.summary()
    schema.plot(Estm.CNN.value)

    estimator = ConvenientNN(schema.getModel())
    estimator.compile(loss=losses.categorical_crossentropy,
                      optimizer='adadelta',
                      metric=['acc'])
    history = estimator.fit(
        db, callbacks=[callbacks.EarlyStopping(patience=20)])

    history.plot(estimator.name+'_'+data)

    print(estimator.evaluate(db.X_train, db.Y_train()))
    y_pred = np.argmax(estimator.predict(db.X_train), axis=-1)
    print(Utils.classificationReport('train', db.y_train, y_pred))

    print(estimator.evaluate(db.X_test, db.Y_test()))
    y_pred = np.argmax(estimator.predict(db.X_test), axis=-1)
    print(Utils.classificationReport('test', db.y_test, y_pred))

    schema.saveWeights(estimator.name+'_'+data)
    schema.extract(estimator.name, db)

    # %% SiameseDouble
    schema = Loader.getSchema(db, version, Estm.SiD)
    schema.summary()
    schema.plot(Estm.SiD.value)

    estimator = SiameseDouble(schema.getModel())
    estimator.build(db.get_shape())
    estimator.compile(loss=contrastive_loss(margin=1),
                      optimizer='adadelta',
                      metric=[metrics.mae, metrics.binary_accuracy])
    history = estimator.fit(db)

    history.plot(estimator.name+'_'+data)

    print(estimator.evaluate(db.X_train, db.y_train, db.info['n_cls']))
    print(estimator.evaluate(db.X_test, db.y_test, db.info['n_cls']))

    schema.saveWeights(estimator.name+'_'+data)
    schema.extract(estimator.name, db)

    # %% SiameseTriplet
    schema = Loader.getSchema(db, version, Estm.SiT)
    schema.summary()
    schema.plot(Estm.SiT.value)

    estimator = SiameseTriplet(schema.getModel())
    estimator.build(db.get_shape())
    estimator.compile(loss=triplet_loss(alpha=0.2),
                      optimizer='adadelta')
    history = estimator.fit(db)

    history.plot(estimator.name+'_'+data)

    print(estimator.evaluate(db.X_train, db.y_train, db.info['n_cls']))
    print(estimator.evaluate(db.X_test, db.y_test, db.info['n_cls']))

    schema.saveWeights(estimator.name+'_'+data)
    schema.extract(estimator.name, db)