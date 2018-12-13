
from Runner import Run
from Reporter import Report

from Config import METHODS
from Config import DATASETS

from Database import INFORM
from Database import load_data
from Database.Utils import reshape
from Database.Utils import get_fewshot
from Database.Utils import plot_histogram

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K


# %% Main Program
if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    rpt = Report()

    for db, db_opt in DATASETS.items():
        n_cls = INFORM[db]['n_cls']
        shape = INFORM[db]['shape']
        X_train, X_test, y_train, y_test = load_data(db)
        X_train = reshape(X_train/255.0, shape)
        X_test = reshape(X_test/255.0, shape)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train)
        plot_histogram(y_train, db+'_train')

        for shot in db_opt['shots']:
            _X_train, _y_train = get_fewshot(X_train, y_train, shot=shot)
            data = (_X_train, X_valid, _y_train, y_valid)

            for bld, bld_opt in METHODS.items():
                # With Augmentation
                rpt.write_dataset(db).write_shot(shot).flush()
                Run(rpt, bld, n_cls, shape, db_opt, bld_opt,
                    *data, X_test, y_test, db, shot, True)
                # Without Augmentation
                rpt.write_dataset(db).write_shot(shot).flush()
                Run(rpt, bld, n_cls, shape, db_opt, bld_opt,
                    *data, X_test, y_test, db, shot, False)
    rpt.flush()
    rpt.close()
