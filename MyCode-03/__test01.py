from Database import INFO
from Database.Utils import reshape
from Database.Utils import load_data
from Database.Utils import get_fewshot

from Schema.Utils import load_schema

from Config import METHOD
from Config import FEWSHOT

from Utils import load_loss
from Utils import load_datagen
from Utils import report_classification

import numpy as np

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TerminateOnNaN

# %% Initialization
verbose = 2
n_jobs = workers = 8
epochs = 1000
batch_size = 128
callbacks = [EarlyStopping(patience=50),
             TerminateOnNaN()]
optimizer = 'adadelta'
builds = sorted(list(METHOD.keys()))
RUNS = [
    {
        'dataset': 'mnist',
        'schema': 'V01',
    },
    {
        'dataset': 'fashion',
        'schema': 'V01',
    },
    {
        'dataset': 'cifar10',
        'schema': 'V03',
    },
    {
        'dataset': 'svhn',
        'schema': 'V03',
    },
    # {
    #     'dataset': 'omniglot',
    #     'schema': 'V01',
    # },
    # {
    #     'dataset': 'mingnet',
    #     'schema': 'V04',
    # },
]

# %% Main Program
for run in RUNS:

    db = run['dataset']

    data = load_data(db)
    n_cls = INFO[db]['n_cls']
    shape = INFO[db]['shape']

    for shot, way in FEWSHOT:
        X_train, X_test, y_train, y_test = get_fewshot(*data, shot, way)

        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        X_train = reshape(X_train / 255.0, shape)
        X_test = reshape(X_test / 255.0, shape)

        for build in builds:
            print(run, shot, way, build)
            schema = load_schema(run['schema'])
            getattr(schema, 'build'+build)(shape, n_cls)

            schema.model.summary()

            loss = load_loss(METHOD[build]['loss'], n_cls)
            schema.model.compile(
                loss=loss, optimizer=optimizer,
                metrics=METHOD[build]['metrics'])

            if METHOD[build]['datagen'].lower() == 'original':
                history = schema.model.fit(
                    X_train, to_categorical(y_train, n_cls), epochs=epochs,
                    batch_size=batch_size, callbacks=callbacks, verbose=verbose,
                    validation_data=(X_test, to_categorical(y_test, n_cls)))
            else:
                datagen = load_datagen(METHOD[build]['datagen'])
                traingen = datagen(X_train, y_train, n_cls, batch_size)
                validgen = datagen(X_test, y_test, n_cls, batch_size)

                history = schema.model.fit_generator(
                    traingen, epochs=epochs, validation_data=validgen, callbacks=callbacks,
                    verbose=verbose, workers=workers, use_multiprocessing=True)

            embed_train = schema.getModel().predict(X_train)
            embed_test = schema.getModel().predict(X_test)

            if 'classification' in METHOD[build].keys():
                if METHOD[build]['classification'] == '':
                    y_score = schema.model.predict(X_test)
                    report_classification(y_test, y_score, n_cls, '_test')
                else:
                    model = getattr(schema, METHOD[build]['classification'])
                    y_score = model.predict(X_test)
                    report_classification(y_test, y_score, n_cls, '_test')

            clf = KNeighborsClassifier(n_neighbors=1, n_jobs=n_jobs)
            clf.fit(embed_train, y_train)
            y_score = clf.predict_proba(embed_test)
            report_classification(y_test, y_score, n_cls, 'title')
            y_pred = np.argmax(y_score, axis=-1)
            print(metrics.accuracy_score(y_test, y_pred))
            print(metrics.f1_score(y_test, y_pred, average='weighted'))