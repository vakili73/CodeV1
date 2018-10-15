# pylint: disable=E0601
from . import BaseEstimator

from Report import History
from Database import Dataset

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers
from pandas.util.testing import isiterable

# %% SiameseTriplet calss


class SiameseTriplet(BaseEstimator):
    def __init__(self,
                 model: Model = Model()):
        super().__init__('SiameseTriplet', model)
        pass

    def _get_batch(self, X_train, y_train, n_cls, batch_size=128):
        train = []
        indices = [np.where(y_train == i)[0] for i in range(n_cls)]
        min_len = [indices[i].size for i in range(n_cls)]
        for _ in range(batch_size):
            classes_order = np.random.permutation(range(n_cls))
            anchor = classes_order[0]
            positive = classes_order[0]
            negetive = np.random.choice(classes_order[1:])
            a_index = np.random.randint(min_len[anchor])
            p_index = np.random.randint(min_len[positive])
            n_index = np.random.randint(min_len[negetive])
            train.append((X_train[indices[anchor][a_index]],
                          X_train[indices[positive][p_index]],
                          X_train[indices[negetive][n_index]]))
        anchor, positive, negetive = zip(*train)
        return np.stack(anchor), np.stack(positive), np.stack(negetive)

    def build(self, shape):
        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)
        processed_a = self.model(input_a)
        processed_p = self.model(input_p)
        processed_n = self.model(input_n)
        concatenate = layers.concatenate([processed_a, processed_p, processed_n])
        self.model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
        pass

    def fit(self, db: Dataset,
            epochs: int = 1000001,
            batch_size: int = 128,
            patience: int = 20) -> History:
        history = {'loss': [],
                   'val_loss': []}
        epoch = []
        for i in range(epochs):
            anchor, positive, negetive = self._get_batch(
                db.X_train, db.y_train, db.info['n_cls'], batch_size)
            loss = self.model.train_on_batch(
                [anchor, positive, negetive], anchor)
            history['loss'].append(loss)
            anchor, positive, negetive = self._get_batch(
                db.X_test, db.y_test, db.info['n_cls'], batch_size)
            val_loss = self.model.test_on_batch(
                [anchor, positive, negetive], anchor)
            history['val_loss'].append(val_loss)
            epoch.append(i)
            if not i % 100:
                print("Batch %d --> loss: %.5f - val_loss: %.5f" %
                      (i, loss, val_loss))
                if i and history['val_loss'][-101] <= val_loss:
                    patience -= 1
                    if not patience:
                        break
        return History(epoch=epoch, history=history)

    def evaluate(self, x, y, n_cls, verbose=2):
        anchor, positive, negetive = self._get_batch(
            x, y, n_cls, 128*100)
        metrics_value = self.model.evaluate([anchor, positive, negetive],
                                            anchor, verbose=verbose)
        metrics_names = self.model.metrics_names
        if isiterable(metrics_value):
            return list(zip(metrics_names, metrics_value))
        else:
            return metrics_names[0], metrics_value

    pass
