
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
        data = load_data(db)
        n_cls = INFORM[db]['n_cls']
        shape = INFORM[db]['shape']
        plot_histogram(data[2], db+'_train')

        for shot in db_opt['shots']:
            X_train, X_test, y_train, y_test = \
                get_fewshot(*data, shot=shot)
            X_train = reshape(X_train/255.0, shape)
            X_test = reshape(X_test/255.0, shape)
            data_tv = train_test_split(
                X_train, y_train, test_size=0.25, stratify=y_train)

            for bld, bld_opt in METHODS.items():
                # With Augmentation
                rpt.write_dataset(db).write_shot(shot).flush()
                Run(rpt, bld, n_cls, shape, db_opt, bld_opt,
                    *data_tv, X_test, y_test, True)
                # Without Augmentation
                rpt.write_dataset(db).write_shot(shot).flush()
                Run(rpt, bld, n_cls, shape, db_opt, bld_opt,
                    *data_tv, X_test, y_test, False)
