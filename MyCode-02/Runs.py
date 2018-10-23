# import numpy as np

from Schema.Utils import load_schema
from Schema.Utils import plot_schema
from Schema.Utils import save_weights
from Schema.Utils import save_feature

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_loss(loss: str):
    if loss.startswith('K-'):
        module = __import__('import tensorflow.keras.losses')
        loss = getattr(module, loss[2:])
    elif loss.startswith('L-'):
        module = __import__('import Losses')
        loss = getattr(module, loss[2:])()
    return loss


def load_datagen(datagen):
    module = __import__('import Generator')
    datagen = getattr(module, datagen)
    return datagen


def MethodNN(name, X_train, X_test, y_train, y_test, n_cls, shape, schema, detail,
             dgen_opt, epochs, batch_size, optimizer, callbacks, augment, prefix):
    schema = load_schema(schema)
    getattr(schema, 'build'+name)(shape, n_cls)
    plot_schema(schema.model, prefix)

    print(prefix)
    schema.model.summary()

    loss = load_loss(detail['loss'])
    schema.model.compile(loss=loss, optimizer=optimizer,
                         metrics=detail['metrics'])

    if augment:
        if detail['datagen'].lower() == 'original':
            datagen = ImageDataGenerator(**dgen_opt)
            datagen.fit(X_train)
            traingen = datagen.flow(X_train, to_categorical(y_train),
                                    batch_size=batch_size)
            validgen = datagen.flow(X_test, to_categorical(y_test),
                                    batch_size=batch_size)
        else:
            datagen = load_datagen('Aug'+detail['datagen'])
            traingen = datagen(X_train, y_train, n_cls, dgen_opt, batch_size)
            validgen = datagen(X_test, y_test, n_cls, dgen_opt, batch_size)

        history = schema.model.fit_generator(traingen, epochs=epochs,
                                             validation_data=validgen,
                                             callbacks=callbacks)
    else:
        if detail['datagen'].lower() == 'original':
            history = schema.model.fit(X_train, to_categorical(y_train),
                                       epochs=epochs, callbacks=callbacks,
                                       validation_data=(X_test, to_categorical(y_test)))
        else:
            datagen = load_datagen(detail['datagen'])
            traingen = datagen(X_train, y_train, n_cls, batch_size)
            validgen = datagen(X_test, y_test, n_cls, batch_size)

            history = schema.model.fit_generator(traingen, epochs=epochs,
                                                 validation_data=validgen,
                                                 callbacks=callbacks)
    
    print(schema.model.metrics_names)
    print('train performance')
    print(schema.model.evaluate(X_train, to_categorical(y_train)))
    print('test performance')
    print(schema.model.evaluate(X_test, to_categorical(y_test)))

    save_weights(schema.model, prefix)

    embed_train = schema.getModel().predict(X_train)
    embed_test = schema.getModel().predict(X_test)

    save_feature(embed_train, y_train, prefix+'_'+schema.extract_layer+'_train')
    save_feature(embed_test, y_test, prefix+'_'+schema.extract_layer+'_test')

    return embed_train, embed_test, history


def MethodKNN(embed_train, embed_test, y_train, y_test, loop_knn):
    raise NotImplementedError


# %% OLD


def MethodCNN(db, shot, n_cls, shape, schema, dgen_opt, X_train, X_test,
              y_train, y_test, epochs, optimizer, callbacks, batch_size=128):
    method = 'CNN'
    schema.buildConventional(shape, n_cls)
    print('Conventional Neural Network Schema Created.')
    plot_schema(schema.model, schema.name+'_build_'+method)
    schema.model.summary()

    print('train without data augmentation...')
    schema.model.compile(loss='categorical_crossentropy',
                         optimizer=optimizer, metrics=['acc'])
    history = schema.model.fit(X_train, to_categorical(y_train),
                               epochs=epochs, callbacks=callbacks,
                               validation_data=(X_test, to_categorical(y_test)))
    file_name = "%s_%sS_%sW_%s" % (
        db, str(shot), str(n_cls), schema.name+'_build_'+method)
    RunUtils.plot_lr_curve(history, file_name)

    y_score = schema.model.predict(X_test)
    y_pred = np.argmax(y_score, axis=-1)

    print('train performance')
    print(schema.model.metrics_names)
    print(schema.model.evaluate(X_train, to_categorical(y_train)))
    print('test performance')
    print(schema.model.evaluate(X_test, to_categorical(y_test)))
    print(classification_report(y_test, y_pred, digits=5))

    RunUtils.plot_roc_curve(file_name, to_categorical(y_test), y_score, n_cls)
    cm = confusion_matrix(y_test, y_pred)
    RunUtils.plot_confusion_matrix(cm, file_name, np.unique(y_train))

    save_weights(schema.model, file_name)

    print('embeding feature extraction...')
    embed_train = schema.getModel().predict(X_train)
    embed_test = schema.getModel().predict(X_test)

    print('feature saving...')
    save_feature(embed_train, y_train, file_name+'_train')
    save_feature(embed_test, y_test, file_name+'_test')

    RunUtils.plot_pca_reduction(embed_test, y_test, file_name+'_pca')
    RunUtils.plot_lsa_reduction(embed_test, y_test, file_name+'_lsa')
    RunUtils.plot_lda_reduction(embed_test, y_test, file_name+'_lda')

    print('train with data augmentation...')
    datagen = ImageDataGenerator(**dgen_opt)
    datagen.fit(X_train)

    train_generator = datagen.flow(X_train, to_categorical(y_train),
                                   batch_size=batch_size)
    test_generator = datagen.flow(X_test, to_categorical(y_test),
                                  batch_size=batch_size)

    schema.buildConventional(shape, n_cls)
    schema.model.compile(loss='categorical_crossentropy',
                         optimizer=optimizer, metrics=['acc'])
    history = schema.model.fit_generator(train_generator,
                                         epochs=epochs, callbacks=callbacks,
                                         validation_data=test_generator)
    file_name += '_Augmented'
    RunUtils.plot_lr_curve(history, file_name)

    y_score = schema.model.predict(X_test)
    y_pred = np.argmax(y_score, axis=-1)

    print('train performance')
    print(schema.model.metrics_names)
    print(schema.model.evaluate(X_train, to_categorical(y_train)))
    print('test performance')
    print(schema.model.evaluate(X_test, to_categorical(y_test)))
    print(classification_report(y_test, y_pred, digits=5))

    RunUtils.plot_roc_curve(file_name, to_categorical(y_test), y_score, n_cls)
    cm = confusion_matrix(y_test, y_pred)
    RunUtils.plot_confusion_matrix(cm, file_name, np.unique(y_train))

    save_weights(schema.model, file_name)

    print('embeding feature extraction...')
    embed_train = schema.getModel().predict(X_train)
    embed_test = schema.getModel().predict(X_test)

    print('feature saving...')
    save_feature(embed_train, y_train, file_name+'_train')
    save_feature(embed_test, y_test, file_name+'_test')

    RunUtils.plot_pca_reduction(embed_test, y_test, file_name+'_pca')
    RunUtils.plot_lsa_reduction(embed_test, y_test, file_name+'_lsa')
    RunUtils.plot_lda_reduction(embed_test, y_test, file_name+'_lda')
