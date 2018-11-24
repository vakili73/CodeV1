# pylint: disable=E0601
from . import BaseEstimator

from Report import History
from Database import Dataset
from Generator import BaseGenerator

from tensorflow.keras import Model, layers
from pandas.util.testing import isiterable

from tensorflow.keras.utils import to_categorical

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

    def fit_on_batch(self,  db: Dataset,
                     gen: BaseGenerator,
                     epochs: int = 1000000,
                     batch_size: int = 128,
                     patience: int = 100,
                     verbose: int = 2,
                     callbacks: list = []) -> History:
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
            X_data, y_data = gen.get_batch()
            metrics_value = self.model.train_on_batch(
                X_data, to_categorical(y_data, num_classes=db.info['n_cls']))
            _update_history('train', metrics_value)
            val_metrics_value = self.model.test_on_batch(db.X_test, db.Y_test())
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

    def evaluate(self, x, y, verbose=2):
        metrics_value = self.model.evaluate(x=x, y=y, verbose=verbose)
        metrics_names = self.model.metrics_names
        if isiterable(metrics_value):
            return list(zip(metrics_names, metrics_value))
        else:
            return metrics_names[0], metrics_value

    pass
