
from .BaseSchema import BaseSchema

from tensorflow.keras import layers, Sequential


class SchemaV02(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV02')
        pass

    def buildConvenient(self, shape, n_cls):
        model = self.build(shape)
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)
        model.add(layers.Dropout(0.2))
        layer02 = layers.Dense(n_cls, activation='softmax')
        model.add(layer02)

        self._add_layer_ex('dense_128_relu', layer01.output)
        self._add_layer_ex('dense_ncls_softmax', layer02.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def buildSiamese(self, shape, n_cls):
        model = self.build(shape)
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def build(self, shape):
        model = Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        return model

    pass
