
from .BaseSchema import BaseSchema

from tensorflow.keras import layers, Sequential


class SchemaV03(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV03')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.5))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)
        model.add(layers.Dropout(0.5))
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
        model.add(layers.Dropout(0.5))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass


    def buildTriplet(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.5))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def build(self, shape):
        model = Sequential()
        model.add(layers.Dense(4096, activation='relu', input_shape=shape))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        return model

    pass
