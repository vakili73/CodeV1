# pylint: disable=E0601
import numpy as np
from Report import History
from Database import Dataset
from tensorflow.keras import Model
from Estimator import BaseEstimator
from pandas.util.testing import isiterable

# %% SiameseDouble calss


class SiameseDouble(BaseEstimator):
    def __init__(self,
                 model: Model = Model()):
        super().__init__('SiameseDouble', model)
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
            patience: int = 20,
            callbacks: list = []) -> History:
        history = {'loss': [],
                   'val_loss': []}
        epoch = []
        for i in range(epochs):
            in_1, in_2, out = self._get_batch(db.X_train, db.y_train,
                                              db.info['n_cls'], batch_size)
            loss = self.model.train_on_batch([in_1, in_2], out)
            history['loss'].append(loss)
            in_1, in_2, out = self._get_batch(db.X_test, db.y_test,
                                              db.info['n_cls'], batch_size)
            val_loss = self.model.test_on_batch([in_1, in_2], out)
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

    def compile(self, loss, optimizer, metric):
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metric)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, n_cls):
        in_1, in_2, out = self._get_batch(x, y, n_cls, 128*100)
        metrics_value = self.model.evaluate([in_1, in_2], out)
        metrics_names = self.model.metrics_names
        if isiterable(metrics_value):
            return list(zip(metrics_names, metrics_value))
        else:
            return metrics_names[0], metrics_value

    pass
