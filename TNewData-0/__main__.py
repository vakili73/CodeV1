# pylint: disable=W0614
from Runs import runs

import Utils
import numpy as np

from Loader import getDataset
from Utils import preProcessing

# %% Main program

for data, run in runs.items():
    db = getDataset(data)
    db = preProcessing(db)
    db.histogram()
    db.summary()

    for estimator, modelSchema, loss, optimizer, metric, callback in run:

        if estimator.__name__ == 'ConvenientNN':

            # model creation
            modelSchema.build(db.get_shape(), db.info['n_cls'])
            modelSchema.summary()
            modelSchema.plot()

            # fitting and training
            estimator = estimator(modelSchema.getModel())
            estimator.compile(loss, optimizer, metric)
            history = estimator.fit(db, callbacks=callback)

            # plot learning curve
            history.plot(estimator.name+'_'+data)

            # model evaluating
            print(estimator.evaluate(db.X_train, db.Y_train()))
            y_pred = np.argmax(estimator.predict(db.X_train), axis=-1)
            print(Utils.classificationReport('train', db.y_train, y_pred))

            print(estimator.evaluate(db.X_test, db.Y_test()))
            y_pred = np.argmax(estimator.predict(db.X_test), axis=-1)
            print(Utils.classificationReport('test', db.y_test, y_pred))

            # model saving
            modelSchema.saveWeights(estimator.name+'_'+data)

            # feature extracting
            modelSchema.extract(estimator.name, db)

        elif estimator.__name__ in ['SiameseDouble', 'SiameseTriplet']:

            # model creation
            modelSchema.build(db.get_shape(), db.info['n_cls'])
            modelSchema.summary()
            modelSchema.plot()

            # fitting and training
            estimator = estimator(modelSchema.getModel())
            estimator.compile(loss, optimizer, metric)
            history = estimator.fit(db, callbacks=callback)

            # plot learning curve
            history.plot(estimator.name+'_'+data)

            # model evaluating
            print(estimator.evaluate(db.X_train, db.y_train, db.info['n_cls']))

            print(estimator.evaluate(db.X_test, db.y_test, db.info['n_cls']))

            # model saving
            modelSchema.saveWeights(estimator.name+'_'+data)

            # feature extracting
            modelSchema.extract(estimator.name, db)
