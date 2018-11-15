import re
import importlib
import numpy as np

from Utils import plot_roc_curve
from Utils import plot_confusion_matrix

from Schema.Utils import load_schema
from Schema.Utils import plot_schema
from Schema.Utils import save_weights
from Schema.Utils import save_feature

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_loss(loss: str, n_cls):
    if loss.startswith('K-'):
        loss = getattr(losses, loss[2:])
    elif loss.startswith('L-'):
        module = __import__('Losses')
        loss = getattr(module, loss[2:])()
    elif loss.startswith('LN-'):
        module = __import__('Losses')
        loss = getattr(module, loss[3:])(n_cls)
    return loss


def load_datagen(datagen):
    module = __import__('Generator')
    datagen = getattr(module, datagen)
    return datagen


def report_classification(y_true, y_score, n_cls, title):
    print(title)
    y_pred = np.argmax(y_score, axis=-1)
    print(classification_report(y_true, y_pred, digits=5))
    plot_roc_curve(title, to_categorical(y_true, n_cls), y_score, n_cls)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, title, np.unique(y_true))


def MethodNN(name, X_train, X_test, y_train, y_test, n_cls, shape, schema, detail,
             dgen_opt, augment, prefix, optimizer, callbacks, batch_size, epochs,
             verbose, use_multiprocessing, workers):
    schema = load_schema(schema)
    getattr(schema, 'build'+name)(shape, n_cls)
    print(prefix)
    schema.model.summary()
    idx = re.search(r'_(None|[0-9]+)Shot_(-1|[0-9]+)Way', prefix).regs[0]
    plot_schema(schema.model, prefix[:idx[0]]+prefix[idx[1]:])

    loss = load_loss(detail['loss'], n_cls)
    schema.model.compile(loss=loss, optimizer=optimizer,
                         metrics=detail['metrics'])
    if augment:
        if detail['datagen'].lower() == 'original':
            datagen = ImageDataGenerator(**dgen_opt)
            datagen.fit(X_train)
            traingen = datagen.flow(X_train, to_categorical(y_train, n_cls),
                                    batch_size=batch_size)
            validgen = datagen.flow(X_test, to_categorical(y_test, n_cls),
                                    batch_size=batch_size)
        else:
            datagen = load_datagen('Aug'+detail['datagen'])
            traingen = datagen(X_train, y_train, n_cls, dgen_opt, batch_size)
            validgen = datagen(X_test, y_test, n_cls, dgen_opt, batch_size)

        history = schema.model.fit_generator(
            traingen, epochs=epochs, validation_data=validgen, callbacks=callbacks,
            verbose=verbose, workers=workers, use_multiprocessing=use_multiprocessing)
    else:
        if detail['datagen'].lower() == 'original':
            history = schema.model.fit(X_train, to_categorical(y_train, n_cls), epochs=epochs,
                                       batch_size=batch_size, callbacks=callbacks, verbose=verbose,
                                       validation_data=(X_test, to_categorical(y_test, n_cls)))
        else:
            datagen = load_datagen(detail['datagen'])
            traingen = datagen(X_train, y_train, n_cls, batch_size)
            validgen = datagen(X_test, y_test, n_cls, batch_size)

            history = schema.model.fit_generator(
                traingen, epochs=epochs, validation_data=validgen, callbacks=callbacks,
                verbose=verbose, workers=workers, use_multiprocessing=use_multiprocessing)

    save_weights(schema.model, prefix)

    embed_train = schema.getModel().predict(X_train)
    embed_test = schema.getModel().predict(X_test)

    if 'classification' in detail.keys():
        if detail['classification'] == '':
            y_score = schema.model.predict(X_train)
            report_classification(y_train, y_score, n_cls, prefix+'_train')
            y_score = schema.model.predict(X_test)
            report_classification(y_test, y_score, n_cls, prefix+'_test')
        else:
            model = getattr(schema, detail['classification'])
            y_score = model.predict(X_train)
            report_classification(y_train, y_score, n_cls, prefix+'_train')
            y_score = model.predict(X_test)
            report_classification(y_test, y_score, n_cls, prefix+'_test')


    save_feature(embed_train, y_train, prefix +
                 '_'+schema.extract_layer+'_train')
    save_feature(embed_test, y_test, prefix +
                 '_'+schema.extract_layer+'_test')

    return embed_train, embed_test, history


def MethodKNN(rpt, X_train, X_test, y_train, y_test, n_cls,
              weights, n_neighbors, n_jobs, prefix):
    def _method_knn(X_data, y_data, title):
        clf = KNeighborsClassifier(
            weights=weights, n_neighbors=n_neighbors, n_jobs=n_jobs)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_data)
        report_classification(y_data, y_score, n_cls, title)
        y_pred = np.argmax(y_score, axis=-1)
        accu_score = metrics.accuracy_score(y_data, y_pred)
        f1_score = metrics.f1_score(y_data, y_pred, average='weighted')
        rpt.write_knn_metrics(weights, n_neighbors, accu_score, f1_score)
    _method_knn(X_test, y_test, prefix+'_test')
