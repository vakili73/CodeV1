import re

from Runs import MethodNN
from Runs import MethodKNN

from Utils import plot_lr_curve
from Utils import plot_reduction

from Config import KNN
from Config import CONFIG
from Config import METHOD
from Config import FEWSHOT

from Database import INFO
from Database.Utils import reshape
from Database.Utils import load_data
from Database.Utils import get_fewshot
from Database.Utils import plot_histogram

from sklearn.utils import shuffle

from tensorflow.keras.callbacks import EarlyStopping

# %% Main Program

n_jobs = 4
options = {
    'optimizer': 'adadelta',
    'callbacks': [EarlyStopping(patience=10)],
    'batch_size': 128,
    'epochs': 1000,
    'verbose': 1,
}

for dataset, schema, dgen_opt in CONFIG:
    loop_data = dataset+'_Schema'+schema
    data = load_data(dataset)

    for shot, way in FEWSHOT:
        loop_fewshot = loop_data+'_%sShot_%sWay' % (str(shot), str(way))
        X_train, X_test, y_train, y_test = get_fewshot(*data, shot, way)

        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        shape = INFO[dataset]['shape']
        X_train = reshape(X_train / 255.0, shape)
        X_test = reshape(X_test / 255.0, shape)

        idx = re.search(r'_SchemaV[0-9]+', loop_fewshot).regs[0]
        plot_histogram(
            y_train, loop_fewshot[:idx[0]]+loop_fewshot[idx[1]:]+'_train')
        plot_histogram(
            y_test, loop_fewshot[:idx[0]]+loop_fewshot[idx[1]:]+'_test')

        for name, detail in METHOD.items():
            n_cls = INFO[dataset]['n_cls'] if way == -1 else way
            db = X_train, X_test, y_train, y_test, n_cls

            # Run without augmentation
            loop_method = loop_fewshot+'_'+name
            embed_train, embed_test, history = MethodNN(
                name, *db, shape, schema, detail, dgen_opt, False, loop_method, **options)
            plot_lr_curve(history, loop_method)
            plot_reduction(embeddings=embed_train,
                           targets=y_train, title=loop_method+'_train')
            plot_reduction(embeddings=embed_test, targets=y_test,
                           title=loop_method+'_test')

            for weights, n_neighbors in KNN:
                loop_knn = loop_method + \
                    '_KNN_{}_{}N'.format(weights, n_neighbors)
                MethodKNN(embed_train, embed_test, y_train, y_test,
                          n_cls, weights, n_neighbors, n_jobs, loop_knn)

            # Run with data augmentation
            if shot != None or dataset == 'omniglot':
                loop_method = loop_fewshot+'_'+name+'_Augmented'
                embed_train, embed_test, history = MethodNN(
                    name, *db, shape, schema, detail, dgen_opt, False, loop_method, **options)
                plot_lr_curve(history, loop_method)
                plot_reduction(embeddings=embed_train,
                               targets=y_train, title=loop_method+'_train')
                plot_reduction(embeddings=embed_test, targets=y_test,
                               title=loop_method+'_test')

                for weights, n_neighbors in KNN:
                    loop_knn = loop_method + \
                        '_KNN_{}_{}N'.format(weights, n_neighbors)
                    MethodKNN(embed_train, embed_test, y_train, y_test,
                              n_cls, weights, n_neighbors, n_jobs, loop_knn)
