
from .BaseSchema import BaseSchema

from tensorflow.keras import layers, Sequential


class SchemaV05(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV05')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.1))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)
        model.add(layers.Dropout(0.1))
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
        model.add(layers.Dropout(0.1))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def buildTriplet(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.1))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def build(self, shape):
        model = Sequential()
        model.add(layers.Conv2D(256, (3, 3), padding='valid',
                                input_shape=shape, activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, activation='relu'))
        return model

    pass
