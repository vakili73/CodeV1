from tensorflow.keras import Model


# %% Base schema class

class BaseSchema(object):
    def __init__(self):
        self.input = NotImplemented
        self.output = NotImplemented
        self.model = Model()
        pass

    def build(self, *args, **kwargs) -> Model:
        raise NotImplementedError

    def getModel(self) -> Model or list:
        if isinstance(self.output, list):
            models = []
            for _output in self.output:
                models.append(Model(self.input, _output))
        return Model(self.input, self.output)

    pass
