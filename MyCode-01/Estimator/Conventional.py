# pylint: disable=E0601
from . import BaseEstimator

from Report import History
from Database import Dataset

from tensorflow.keras import Model, layers
from pandas.util.testing import isiterable

# %% Conventional calss


class Conventional(BaseEstimator):
    def __init__(self,
                 model: Model = Model()):
        super().__init__('Conventional', model)
        pass

    def fit(self, db: Dataset,
            epochs: int = 1000,
            batch_size: int = 128,
            verbose: int = 2,
            callbacks: list = []) -> History:
        history = self.model.fit(db.X_train, db.Y_train(),
                                 validation_data=(db.X_test, db.Y_test()),
                                 epochs=epochs, batch_size=batch_size,
                                 verbose=verbose, callbacks=callbacks)
        return History(history.epoch, history.params, history.history)

    def evaluate(self, x, y, verbose=2):
        metrics_value = self.model.evaluate(x=x, y=y, verbose=verbose)
        metrics_names = self.model.metrics_names
        if isiterable(metrics_value):
            return list(zip(metrics_names, metrics_value))
        else:
            return metrics_names[0], metrics_value

    pass
