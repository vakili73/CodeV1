from tensorflow.keras import Model


# %% Base schema class

class BaseSchema(object):
    def __init__(self,
                 name: str = ''):
        self.name = name
        self.input = NotImplemented
        self.output = NotImplemented
        self.model: Model = Model()
        pass

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def getModel(self) -> Model:
        return Model(self.input, self.output)

    pass
