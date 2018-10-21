
import Runs

from Config import CONFIG as RunCONFIG
from Config import FEWSHOT as FewShotCONFIG

from Database import Utils as DbUtils
from Database import CONFIG as DbCONFIG

from Schema import Utils as ScUtils

from tensorflow.keras.callbacks import EarlyStopping


# %% Main program

for db, mdl, dgen_opt in RunCONFIG:

    print('dataset: %s,\t model: %s' % (db, mdl))

    n_cls = DbCONFIG[db]['n_cls']
    shape = DbCONFIG[db]['shape']

    print(DbCONFIG[db])
    print('datagen opt')
    print(dgen_opt)

    X_train, X_test, y_train, y_test = DbUtils.laod_data('mnist')
    print('Data histogram plotting...')
    DbUtils.plot_histogram(y_train, db+'_train')
    DbUtils.plot_histogram(y_test, db+'_test')

    X_train = DbUtils.reshape(X_train / 255.0, shape)
    X_test = DbUtils.reshape(X_test / 255.0, shape)

    schema = ScUtils.load_schema(mdl)

    optimizer = 'adadelta'
    callbacks = EarlyStopping(patience=10)
    epochs = 1000

    for shot, way in FewShotCONFIG:

        print("%s-Shot, %s-Way" % (str(shot), str(way)))
        X_train, X_test, y_train, y_test = DbUtils.get_fewshot(
            X_train, X_test, y_train, y_test, shot, way)
        print('Shotted Data histogram plotting...')
        DbUtils.plot_histogram(y_train, db+'_train_%sS_%sW'
                               % (str(shot), str(way)))
        DbUtils.plot_histogram(y_test, db+'_test_%sS_%sW'
                               % (str(shot), str(way)))

        # Conventional Neural Network
        print('Method: Conventional Neural Network')
        n_cls = n_cls if way == -1 else way
        Runs.MethodCNN(db, shot, n_cls, shape, schema, dgen_opt, X_train, X_test,
                       y_train, y_test, epochs, optimizer, callbacks)
