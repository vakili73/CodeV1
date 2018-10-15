# pylint: disable=E0601
from Report import History
from tensorflow.keras import Model

# %% Abstract estimator


class BaseEstimator(object):
    def __init__(self,
                 name: str = '',
                 model: Model = Model()):
        self.name = name
        self.model = model
        pass

    def fit(self, *args, **kwargs) -> History:
        raise NotImplementedError

    def compile(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    pass


# %% testing
if __name__ == '__main__':
    baseEstimator = BaseEstimator()
    pass
