import numpy as np

import Utils as RunUtils

from Schema.Utils import plot_schema

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report


def MethodCNN(db, shot, n_cls, shape, schema, dgen_opt, X_train, X_test,
              y_train, y_test, epochs, optimizer, callbacks, batch_size=128):
    method = 'CNN'
    schema.buildConventional(shape, n_cls)
    print('Conventional Neural Network Schema Created.')
    plot_schema(schema.model, schema.name+'_build_'+method)
    schema.model.summary()

    print('training with NonAugmented data...')
    schema.model.compile(loss='categorical_crossentropy',
                         optimizer=optimizer, metrics=['acc'])
    history = schema.model.fit(X_train, to_categorical(y_train),
                               epochs=epochs, callbacks=callbacks,
                               validation_data=(X_test, y_test))
    RunUtils.plot_lr_curve(history, "%s_%sS_%sW_%s" % (
        db, str(shot), str(n_cls), schema.name+'_build_'+method))
        
    y_score = schema.model.predict(X_test)
    y_pred = np.argmax(y_score, axis=-1)

    print(classification_report(y_test, y_pred, digits=5))

    raise NotImplementedError
