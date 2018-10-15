# pylint: disable=E0601
from . import BaseEstimator

from Report import History
from Database import Dataset

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers
from pandas.util.testing import isiterable

# %% Siamese calss


class Siamese(BaseEstimator):
    def __init__(self,
                 model: Model = Model()):
        super().__init__('Siamese', model)
        pass

    def build(self, shape):
        def _euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
        input_a = layers.Input(shape=shape)
        input_b = layers.Input(shape=shape)
        processed_a = self.model(input_a)
        processed_b = self.model(input_b)
        distance = layers.Lambda(_euclidean_distance)(
            [processed_a, processed_b])
        self.model = Model(inputs=[input_a, input_b], outputs=distance)
        pass

    def _get_batch(self, X_train, y_train, n_cls, batch_size=128):
        train = []
        indices = [np.where(y_train == i)[0] for i in range(n_cls)]
        min_len = [indices[i].size for i in range(n_cls)]
        for _ in range(batch_size//2):
            classes_order = np.random.permutation(range(n_cls))
            anchor = classes_order[0]
            other = np.random.choice(classes_order[1:])
            o_index = np.random.randint(min_len[other])
            a_index = np.random.randint(min_len[anchor])
            train.append((X_train[indices[other][o_index]],
                          X_train[indices[anchor][a_index]], 1))
            a_index = np.random.randint(min_len[anchor], size=2)
            train.append((X_train[indices[anchor][a_index[0]]],
                          X_train[indices[anchor][a_index[1]]], 0))
        in_1, in_2, out = zip(*train)
        return np.stack(in_1), np.stack(in_2), np.stack(out)

    def fit(self, db: Dataset,
            epochs: int = 1000001,
            batch_size: int = 128,
            patience: int = 20) -> History:
        history = {}
        for item in self.model.metrics_names:
            history.update({item: []})
            history.update({'val_'+item: []})

        def _print_report(ltype, metrics_value):
            i = 0
            for item in self.model.metrics_names:
                if ltype == 'train':
                    print("%s: %.5f - " % (item, metrics_value[i]), end='')
                elif ltype == 'test':
                    print("%s: %.5f - " %
                          ('val_'+item, metrics_value[i]), end='')
                i += 1

        def _update_history(ltype, metrics_value):
            i = 0
            for item in self.model.metrics_names:
                if ltype == 'train':
                    history[item].append(metrics_value[i])
                elif ltype == 'test':
                    history['val_'+item].append(metrics_value[i])
                i += 1
        epoch = []
        for i in range(epochs):
            in_1, in_2, out = self._get_batch(db.X_train, db.y_train,
                                              db.info['n_cls'], batch_size)
            metrics_value = self.model.train_on_batch([in_1, in_2], out)
            _update_history('train', metrics_value)
            in_1, in_2, out = self._get_batch(db.X_test, db.y_test,
                                              db.info['n_cls'], batch_size)
            val_metrics_value = self.model.test_on_batch([in_1, in_2], out)
            _update_history('test', val_metrics_value)
            epoch.append(i)
            if not i % 100:
                print("Batch %d --> " % i, end='')
                _print_report('train', metrics_value)
                _print_report('test', val_metrics_value)
                print('')
                if i and history['val_loss'][-101] <= val_metrics_value[0]:
                    patience -= 1
                    if not patience:
                        break
        return History(epoch=epoch, history=history)

    def evaluate(self, x, y, n_cls, verbose=2):
        in_1, in_2, out = self._get_batch(x, y, n_cls, 128*100)
        metrics_value = self.model.evaluate([in_1, in_2], out,
                                            verbose=verbose)
        metrics_names = self.model.metrics_names
        if isiterable(metrics_value):
            return list(zip(metrics_names, metrics_value))
        else:
            return metrics_names[0], metrics_value

    pass
