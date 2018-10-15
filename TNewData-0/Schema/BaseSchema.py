from tensorflow.keras import layers, Model, Sequential

# %% Base schema class


class BaseSchema(object):
    def __init__(self,
                 name: str = ''):
        self.name = name
        pass

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def getModel(self, *args, **kwargs) -> Model:
        raise NotImplementedError

    def getInput(self, *args, **kwargs):
        raise NotImplementedError

    def getOutput(self, *args, **kwargs):
        raise NotImplementedError

    def saveWeights(self, *args, **kwargs):
        raise NotImplementedError

    def loadWeights(self, *args, **kwargs):
        raise NotImplementedError
        
    def summary(self, *args, **kwargs):
        raise NotImplementedError

    def extract(self, *args, **kwargs):
        raise NotImplementedError
    pass

# %% testing
if __name__ == '__main__':
    baseSchema = BaseSchema()